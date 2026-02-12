import os
import sqlite3
import json
import openslide
import cv2
import numpy as np
from shapely.wkt import loads
from shapely.geometry import Polygon, Point
from pathlib import Path
from collections import defaultdict

# ================= CONFIGURATION =================
# Update these paths to match your actual folder locations
DB_PATH = "/Volumes/1TB HDD 01/CATCH/CATCH.sqlite"             # Path to your SQLite file
WSI_DIR = "/Volumes/1TB HDD 01/CATCH/WSI"                     # Folder containing your .svs files
OUTPUT_DIR = "/Volumes/1TB HDD 01/CATCH4WSISAM(withHard-ve&1024)2"        # Where the output files will be saved

# Training Parameters
PATCH_SIZE = 1024                             # The output size (256x256)
BASE_MAG = 40                                # The native magnification of your .svs files (Level 0)
TARGET_MAG_HIGH = 40                         # High Res Magnification
TARGET_MAG_LOW = 20                          # Low Res Magnification
NEGATIVE_PER_POLYGON = 8

# Number of extra edge patches to take per tumor (in addition to the center)
NUM_EDGE_PATCHES = 6  # Max number of edge patches per tumor
# =================================================

def get_patch_at_mag(slide, center_x, center_y, target_mag, base_mag=40, out_size=1024):
    """
    Extracts a patch centered at (center_x, center_y) from the WSI at a specific magnification.
    Returns the image and the tuple (top_left_x, top_left_y, region_size_at_level_0).
    """
    # Calculate how much area we need to read from Level 0 to get the desired output size
    downsample_factor = base_mag / target_mag
    read_size_lvl0 = int(out_size * downsample_factor)
    
    # Calculate Top-Left coordinate at Level 0
    tl_x = int(center_x - (read_size_lvl0 / 2))
    tl_y = int(center_y - (read_size_lvl0 / 2))
    
    # Read the region from Level 0 (highest resolution)
    # Note: We read from Level 0 to ensure accuracy, then resize down.
    try:
        img = slide.read_region((tl_x, tl_y), 0, (read_size_lvl0, read_size_lvl0)).convert("RGB")
        img = np.array(img)
        
        # Resize to the target output size (256x256)
        if read_size_lvl0 != out_size:
            img = cv2.resize(img, (out_size, out_size), interpolation=cv2.INTER_LINEAR)
            
        return img, (tl_x, tl_y, read_size_lvl0)
        
    except Exception as e:
        print(f"    Error reading region at {tl_x}, {tl_y}: {e}")
        return None, None

def rasterize_mask(polygon, tl_x, tl_y, region_size_lvl0, out_size):
    """
    Converts a Shapely polygon coordinate list into a black-and-white mask image.
    1. Creates a black square.
    2. Shifts polygon coordinates relative to the patch.
    3. Scales coordinates to the output size.
    4. Draws ('paints') the polygon in white.
    """
    # Create empty black mask (0)
    mask = np.zeros((out_size, out_size), dtype=np.uint8)
    
    # Robust Geometry Handling
    if polygon.geom_type in ['MultiPolygon', 'GeometryCollection']:
        # Extract only the Polygons from the collection
        polys = [p for p in polygon.geoms if p.geom_type == 'Polygon']
    else:
        polys = [polygon]
        
    for poly in polys:
        if poly.is_empty: continue
        exterior_coords = np.array(poly.exterior.coords)
    
        # Transform coordinates:
        # 1. Shift: Subtract the top-left corner of the patch
        exterior_coords[:, 0] = exterior_coords[:, 0] - tl_x
        exterior_coords[:, 1] = exterior_coords[:, 1] - tl_y
        
        # 2. Scale: Adjust for the resize (downsampling)
        scale = out_size / region_size_lvl0
        exterior_coords = exterior_coords * scale
        
        # Prepare points for OpenCV (must be int32)
        pts = exterior_coords.astype(np.int32)
        pts = pts.reshape((-1, 1, 2)) # Shape required by fillPoly
        
        # Draw the filled polygon with value 1 (White)
        cv2.fillPoly(mask, [pts], 1)
        
    return mask

def get_sampling_points(polygon, patch_size_lvl0):
    """
    Returns a list of (x, y, type) tuples.
    Type is just for logging: 'boundary' or 'internal'.
    """
    points = []
    
    # Handle MultiPolygons by processing the largest component only (usually the main tumor)
    if isinstance(polygon, MultiPolygon):
        # Pick the polygon with the largest area
        main_poly = max(polygon.geoms, key=lambda p: p.area)
    else:
        main_poly = polygon

    # --- 1. Internal Samples (Non-Edges) ---
    # Representative point is guaranteed to be inside
    p_rep = main_poly.representative_point()
    points.append((p_rep.x, p_rep.y, 'internal'))
    
    # Centroid (might be same as rep, or outside if 'C' shape, but usually good)
    p_cent = main_poly.centroid
    if main_poly.contains(p_cent) and p_cent.distance(p_rep) > patch_size_lvl0:
        points.append((p_cent.x, p_cent.y, 'internal'))

    # --- 2. Boundary Samples (Edges) ---
    perimeter = main_poly.exterior.length
    
    # Determine step size
    # We ideally want a patch every 'Patch Width'
    step_size = patch_size_lvl0
    
    # Calculate how many points that would be
    num_points = int(perimeter / step_size)
    
    # Apply Cap (Don't take 100 points for a huge tumor)
    if num_points > MAX_BOUNDARY_POINTS:
        # Increase step size to fit exactly MAX points
        num_points = MAX_BOUNDARY_POINTS
        step_size = perimeter / num_points
    
    # Minimum 1 point if perimeter is tiny
    if num_points < 1: 
        num_points = 1
        
    # Walk the perimeter
    for i in range(num_points):
        # Interpolate returns a point at distance d along boundary
        dist = i * step_size
        p_edge = main_poly.exterior.interpolate(dist)
        points.append((p_edge.x, p_edge.y, 'boundary'))
        
    return points

def fetch_and_group_annotations(db_path):
    """
    Connects to DB, joins tables, and groups coordinates into Polygons.
    Returns structure: { 'filename.svs': [Polygon1, Polygon2, ...] }
    """
    print(f"Querying Database: {db_path}...")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # JOIN QUERY:
    # 1. Get Filename from Slides
    # 2. Get Class from Labels (Filter >= 7)
    # 3. Get Coordinates (X, Y)
    query = """
        SELECT 
            S.filename,
            L.annoId,
            C.coordinateX,
            C.coordinateY
        FROM Annotations_coordinates C
        JOIN Annotations_label L ON C.annoId = L.annoId
        JOIN Slides S ON C.slide = S.uid
        WHERE L.class >= 7
        ORDER BY S.filename, L.annoId
    """
    
    try:
        cursor.execute(query)
        rows = cursor.fetchall()
    except Exception as e:
        print(f"SQL Error: {e}")
        return {}
    
    print(f"  Fetched {len(rows)} coordinate points. Grouping into Polygons...")

    # Grouping Logic
    # structure: temp_data['slide_name']['anno_id'] = [(x,y), (x,y)...]
    temp_data = defaultdict(lambda: defaultdict(list))
    
    for filename, anno_id, x, y in rows:
        temp_data[filename][anno_id].append((x, y))
        
    # Convert lists of points into Shapely Polygons
    # structure: final_data['slide_name'] = [Polygon, Polygon...]
    final_data = defaultdict(list)
    
    total_polys = 0
    for filename, annotations in temp_data.items():
        for anno_id, coords in annotations.items():
            if len(coords) < 3: continue # Need at least 3 points for a polygon
            
            # Create Polygon from points
            poly = Polygon(coords)
            
            # Validation: Ensure polygon is valid
            if not poly.is_valid:
                poly = poly.buffer(0) # Attempt to fix self-intersections
                
            final_data[filename].append(poly)
            total_polys += 1
            
    print(f"  Reconstructed {total_polys} valid polygons across {len(final_data)} slides.")
    return final_data

def is_tissue(slide, x, y, size_lvl0, threshold=220):
    """
    Checks if a patch contains tissue (is not just empty white glass).
    Reads a tiny thumbnail of the region to check quickly.
    """
    try:
        # Read a tiny version (1x1 pixel is too small, maybe 16x16)
        # We read at a lower level for speed
        region = slide.read_region((int(x), int(y)), 2, (16, 16)).convert("L")
        region_np = np.array(region)
        # If average brightness is less than threshold (220), it's likely tissue.
        # (Glass is usually pure white ~255)
        return np.mean(region_np) < threshold
    except:
        return False



def main():
    # 1. Setup Directories
    for sub in ["high/imgs", "high/masks", "low/imgs", "low/masks"]:
        os.makedirs(os.path.join(OUTPUT_DIR, sub), exist_ok=True)

    # 2. Get Data Structure
    slides_data = fetch_and_group_annotations(DB_PATH)
    
    data_records = []
    global_idx = 0

    # 3. Process Slide by Slide (Efficiency Boost)
    for filename, polygons in slides_data.items():
        
        # Ensure extension
        full_filename = filename if filename.endswith(".svs") else f"{filename}.svs"
        wsi_path = os.path.join(WSI_DIR, full_filename)
        
        if not os.path.exists(wsi_path):
            print(f"  [Warning] Slide not found: {wsi_path}")
            continue

        print(f"Processing {full_filename} ({len(polygons)} annotations)...")
        
        try:
            slide = openslide.OpenSlide(wsi_path)
            
            positive_targets = []

            for poly in polygons:
                # ========================================================
                # 1. Define Targets: Where do we want to cut patches?
                # ========================================================
                
                # A. The Center (Original Logic - Reliable)
                minx, miny, maxx, maxy = poly.bounds
                center_x = (minx + maxx) / 2
                center_y = (miny + maxy) / 2
                positive_targets.append((center_x, center_y, "center"))
                
                # B. The Edges (New Logic - Boundary Sampling)
                # We extract the main polygon ring to walk along the edge
                if poly.geom_type in ['MultiPolygon', 'GeometryCollection']:
                    # Use largest island for edge sampling
                    sub_polys = [p for p in poly.geoms if p.geom_type == 'Polygon']
                    if sub_polys:
                        search_poly = max(sub_polys, key=lambda p: p.area)
                    else:
                        search_poly = None
                else:
                    search_poly = poly

                if search_poly and NUM_EDGE_PATCHES > 0:
                    perimeter = search_poly.exterior.length
                    # Sample 4 equidistant points (0%, 25%, 50%, 75% of perimeter)
                    for i in range(NUM_EDGE_PATCHES):
                        dist = (i / NUM_EDGE_PATCHES) * perimeter
                        pt = search_poly.exterior.interpolate(dist)
                        positive_targets.append((pt.x, pt.y, "edge"))

            # ========================================================
            # PART 2: NEGATIVES (Pure Background) - NEW LOGIC !!
            # ========================================================
            # We want to find N patches that DO NOT overlap with any polygon
            # Target: roughly same amount as centers (e.g., 1 per polygon or fixed number)
            num_negatives_needed = len(polygons) * NEGATIVE_PER_POLYGON  # E.g., 2 negatives per tumor found
            negative_targets = []
            
            w_slide, h_slide = slide.dimensions
            attempts = 0
            max_attempts = 1000 # Don't loop forever
            
            while len(negative_targets) < num_negatives_needed and attempts < max_attempts:
                attempts += 1
                
                # 1. Pick Random Point
                rx = np.random.randint(0, w_slide - PATCH_SIZE)
                ry = np.random.randint(0, h_slide - PATCH_SIZE)
                
                # 2. Check: Is it Tissue? (Don't save empty glass)
                if not is_tissue(slide, rx, ry, PATCH_SIZE):
                    continue
                    
                # 3. Check: Is it inside ANY tumor?
                # Create a Point object for the patch center
                patch_center = Point(rx + PATCH_SIZE//2, ry + PATCH_SIZE//2)
                
                is_clean = True
                for poly in polygons:
                    # Buffer adds a safety margin so we aren't right next to a tumor
                    if poly.buffer(100).contains(patch_center): 
                        is_clean = False
                        break
                
                if is_clean:
                    negative_targets.append((rx, ry, "negative"))

            # Combine all targets
            all_targets = positive_targets + negative_targets
            # ========================================================
            # 2. Extract Patches for all Targets
            # ========================================================
            for (cx, cy, p_type) in all_targets:
                
                # High Res (40x)
                img_h, p_h = get_patch_at_mag(slide, cx, cy, TARGET_MAG_HIGH, BASE_MAG, PATCH_SIZE)
                if img_h is None: continue

                img_l, p_l = get_patch_at_mag(slide, cx, cy, TARGET_MAG_LOW, BASE_MAG, PATCH_SIZE)
                if img_l is None: continue

                mask_h = np.zeros((PATCH_SIZE, PATCH_SIZE), dtype=np.uint8)
                mask_l = np.zeros((PATCH_SIZE, PATCH_SIZE), dtype=np.uint8)

                if p_type == "negative":
                    pass
                else:
                    for poly in polygons:
                        # Draw this specific polygon relative to the current patch coordinates
                        temp_mask_h = rasterize_mask(poly, p_h[0], p_h[1], p_h[2], PATCH_SIZE)
                        temp_mask_l = rasterize_mask(poly, p_l[0], p_l[1], p_l[2], PATCH_SIZE)
                        
                        # Merge into main mask using Maximum (Logical OR)
                        # This ensures if two tumors overlap/touch in the frame, both are white.
                        mask_h = np.maximum(mask_h, temp_mask_h)
                        mask_l = np.maximum(mask_l, temp_mask_l)
                    
                # Save
                name = f"{os.path.splitext(full_filename)[0]}_{global_idx}_{p_type}"
                
                paths = {
                    "high_im": f"{OUTPUT_DIR}/high/imgs/{name}.png",
                    "high_gt": f"{OUTPUT_DIR}/high/masks/{name}.png",
                    "low_im":  f"{OUTPUT_DIR}/low/imgs/{name}.png",
                    "low_gt":  f"{OUTPUT_DIR}/low/masks/{name}.png",
                }
                
                cv2.imwrite(paths["high_im"], cv2.cvtColor(img_h, cv2.COLOR_RGB2BGR))
                cv2.imwrite(paths["high_gt"], mask_h * 255)
                cv2.imwrite(paths["low_im"], cv2.cvtColor(img_l, cv2.COLOR_RGB2BGR))
                cv2.imwrite(paths["low_gt"], mask_l * 255)
                
                data_records.append({
                    "high": {"im_path": paths["high_im"], "gt_path": paths["high_gt"]},
                    "low":  {"im_path": paths["low_im"], "gt_path": paths["low_gt"]}
                })
            
                global_idx += 1
            
            slide.close()
        except Exception as e:
            print(f"Error processing {filename}: {e}")

    with open(f"{OUTPUT_DIR}/dataset.json", "w") as f:
        json.dump(data_records, f, indent=4)
    print(f"Done! Saved {len(data_records)} pairs.")

    #     try:
    #         slide = openslide.OpenSlide(wsi_path)
            
    #         for poly in polygons:
    #             # Get Center
    #             minx, miny, maxx, maxy = poly.bounds
    #             center_x = (minx + maxx) / 2
    #             center_y = (miny + maxy) / 2
                
    #             # --- High Res (40x) ---
    #             img_h, p_h = get_patch_at_mag(slide, center_x, center_y, TARGET_MAG_HIGH, BASE_MAG, PATCH_SIZE)
    #             if img_h is None: continue
    #             mask_h = rasterize_mask(poly, p_h[0], p_h[1], p_h[2], PATCH_SIZE)
                
    #             # --- Low Res (20x) ---
    #             img_l, p_l = get_patch_at_mag(slide, center_x, center_y, TARGET_MAG_LOW, BASE_MAG, PATCH_SIZE)
    #             if img_l is None: continue
    #             mask_l = rasterize_mask(poly, p_l[0], p_l[1], p_l[2], PATCH_SIZE)
                
    #             # --- Save ---
    #             name = f"{os.path.splitext(full_filename)[0]}_{global_idx}"
                
    #             paths = {
    #                 "high_im": f"{OUTPUT_DIR}/high/imgs/{name}.png",
    #                 "high_gt": f"{OUTPUT_DIR}/high/masks/{name}.png",
    #                 "low_im":  f"{OUTPUT_DIR}/low/imgs/{name}.png",
    #                 "low_gt":  f"{OUTPUT_DIR}/low/masks/{name}.png",
    #             }
                
    #             cv2.imwrite(paths["high_im"], cv2.cvtColor(img_h, cv2.COLOR_RGB2BGR))
    #             cv2.imwrite(paths["high_gt"], mask_h * 255)
    #             cv2.imwrite(paths["low_im"], cv2.cvtColor(img_l, cv2.COLOR_RGB2BGR))
    #             cv2.imwrite(paths["low_gt"], mask_l * 255)
                
    #             data_records.append({
    #                 "high": {"im_path": paths["high_im"], "gt_path": paths["high_gt"]},
    #                 "low":  {"im_path": paths["low_im"], "gt_path": paths["low_gt"]}
    #             })
                
    #             global_idx += 1
            
    #         slide.close()
            
    #     except Exception as e:
    #         print(f"  Failed to process slide {filename}: {e}")

    # # 4. Save Index
    # with open(f"{OUTPUT_DIR}/dataset.json", "w") as f:
    #     json.dump(data_records, f, indent=4)
        
    # print(f"Done! Saved {len(data_records)} pairs to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()