#%%

import numpy as np
import time
import os
import tifffile

np.random.seed(42)

# Example data (could be replaced with memory-mapped files)
# Memory-map the arrays for better memory management
def create_memmap_array(shape, filename=None, dtype=np.float32):
    """Create a memory-mapped array for large datasets"""
    if filename is None:
        filename = f"memmap_temp_{os.getpid()}.dat"
    
    memmap_array = np.memmap(filename, dtype=dtype, mode='w+', shape=shape)
    return memmap_array, filename

# Use memory-mapped arrays for large data
# img_roi_perfect, perfect_filename = create_memmap_array((300, 1000, 1000))
# img_roi_perfect[:] = np.random.rand(300, 1000, 1000) * 255
# Uncomment the above and replace the below if you want to use memory mapping

img_roi_perfect = np.random.rand(6, 1000, 1000) * 255
img_roi_imperfect = np.random.rand(5, 9, 9) * 255

img_roi_p_shape = img_roi_perfect.shape[-2:]
img_roi_ip_shape = img_roi_imperfect.shape[-2:]
frame_range = 1, 4
roi_size = (10, 10)

def generate_rois_vectorized(image_shape, roi_size):
    """Generate ROIs using vectorized operations"""
    height, width = image_shape
    roi_w, roi_h = roi_size
    
    # Calculate grid dimensions
    cols = int(np.ceil(width / roi_w))
    rows = int(np.ceil(height / roi_h))

    
    # Create 2D arrays of coordinates for better vectorization
    col_indices, row_indices = np.meshgrid(np.arange(cols), np.arange(rows))
    
    # Convert to 1D arrays
    x_coords = (col_indices.flatten() * roi_w).astype(np.int32)
    y_coords = (row_indices.flatten() * roi_h).astype(np.int32)
    
    # Calculate dimensions with boundary handling (vectorized)
    widths = np.minimum(roi_w, width - x_coords).astype(np.int32)
    heights = np.minimum(roi_h, height - y_coords).astype(np.int32)
    
    # Create ROIs as a structured numpy array for better memory access patterns
    dtype = np.dtype([('x', np.int32), ('y', np.int32), ('w', np.int32), ('h', np.int32)])
    rois = np.zeros(len(x_coords), dtype=dtype)
    rois['x'] = x_coords
    rois['y'] = y_coords
    rois['w'] = widths
    rois['h'] = heights
    
    return rois, cols, rows, (cols,rows)

def define_neighbors_for_all_rois(rois_per_row, rois_per_col):
    """Pre-compute all neighbors for all ROIs at once"""
    total_rois = rois_per_row * rois_per_col
    neighbors_list = []
    
    # Create a 2D grid representation of ROIs
    roi_grid = np.arange(total_rois).reshape(rois_per_col, rois_per_row)
    
    # Define neighbor offsets (excluding self)
    offsets = [(i, j) for i in [-1, 0, 1] for j in [-1, 0, 1] if not (i == 0 and j == 0)]
    
    for roi_idx in range(total_rois):
        row = roi_idx // rois_per_row
        col = roi_idx % rois_per_row
        
        # Find all valid neighbors
        neighbors = []
        for dy, dx in offsets:
            new_row, new_col = row + dy, col + dx
            if 0 <= new_row < rois_per_col and 0 <= new_col < rois_per_row:
                neighbors.append(roi_grid[new_row, new_col])
        
        neighbors_list.append(neighbors)
    
    return neighbors_list

def process_frame_range_optimized(frame, rois):
    """Calculate intensities for all ROIs in a frame with vectorized operations where possible"""
    roi_intensities = np.full(len(rois), fill_value = np.nan, dtype=np.float32)
    
    # Extract ROI data with a more efficient access pattern
    for i in range(len(rois)):
        roi = rois[i]
        x, y, w, h = roi['x'], roi['y'], roi['w'], roi['h']
        
        # Get ROI pixels directly
        roi_pixels = frame[y:y+h, x:x+w]
        non_zero_mask = roi_pixels > 0
        
        if np.any(non_zero_mask):
            roi_intensities[i] = np.mean(roi_pixels[non_zero_mask])
    
    return roi_intensities

def calculate_f0_per_roi_optimized(avg_roi_array, frame_range, neighbors_list):
    """Calculate baseline fluorescence for each ROI with vectorized operations"""
    start_frame, end_frame = frame_range
    frame_indices = np.arange(start_frame, end_frame + 1)
    num_rois = len(neighbors_list)
    
    # Pre-allocate arrays for better memory usage
    f0_per_roi = np.zeros(num_rois, dtype=np.float32)
    
    for roi_idx in range(num_rois):
        neighbors = neighbors_list[roi_idx]
        roi_and_neighbors = neighbors + [roi_idx]

        subarray = avg_roi_array[start_frame:end_frame+1, roi_and_neighbors]
        subarray = np.where(subarray == 0, np.nan, subarray)

        frame_means = np.nanmean(subarray, axis = 1)
        f0_per_roi[roi_idx] = np.nanmean(frame_means)
        
    
    return f0_per_roi

def create_all_roi_masks(rois, img_shape):
    """Pre-compute masks for all ROIs"""
    height, width = img_shape
    all_masks = []
    
    for i in range(len(rois)):
        roi = rois[i]
        x, y, w, h = roi['x'], roi['y'], roi['w'], roi['h']
        
        # Create sparse representation of mask: just store coordinates
        y_indices = slice(y, min(y+h, height))
        x_indices = slice(x, min(x+w, width))
        all_masks.append((y_indices, x_indices))
    
    return all_masks

def dff_pixel_optimized(frame, all_masks, f0_per_roi):
    """Apply dF/F transformation with pre-computed masks and vectorized operations"""
    result_frame = np.zeros_like(frame, dtype=np.float32)
    
    for roi_index, (y_slice, x_slice) in enumerate(all_masks):
        f0 = f0_per_roi[roi_index]
        if f0 <= 0:
            continue
        
        # Get ROI pixels directly using the pre-computed slices
        roi_pixels = frame[y_slice, x_slice]
        
        # Create mask for non-zero pixels
        non_zero_mask = roi_pixels > 0
        
        if np.any(non_zero_mask):
            # Create temporary result array
            temp_result = np.zeros_like(roi_pixels, dtype=np.float32)
            
            # Calculate dF/F only for non-zero pixels
            temp_result[non_zero_mask] = (roi_pixels[non_zero_mask] - f0) / f0
            
            # Assign results back to the frame
            result_frame[y_slice, x_slice] = temp_result
    
    return result_frame

def process_frame_frame_optimized(image_array, roi_size, frame_range, use_memmap=False):
    """Main processing function with optimized operations"""
    frames = image_array.shape[0]
    height, width = image_array.shape[1:]
    
    print("Starting optimized processing")
    
    # Generate ROIs only once at the beginning 
    print("Generating ROIs...")
    rois, cols, rows, rois_shape = generate_rois_vectorized((height, width), roi_size)
    
    # Pre-compute neighbors for all ROIs
    print("Computing neighbors...")
    neighbors_list = define_neighbors_for_all_rois(cols, rows)
    print(neighbors_list)
    
    # Pre-compute masks for all ROIs
    print("Creating ROI masks...")
    all_masks = create_all_roi_masks(rois, (height, width))
    print(all_masks)
    
    # Pre-allocate ROI intensities array instead of using a list
    all_roi_intensities = np.zeros((frames, len(rois)), dtype=np.float32)

    # Process all frames to get ROI intensities
    print("Processing frame intensities...")
    for frame_idx in range(frames):
        frame = image_array[frame_idx]
        all_roi_intensities[frame_idx] = process_frame_range_optimized(frame, rois)
        
        if frame_idx % 10 == 0 or frame_idx == frames-1:
            print(f"Processed roi_intensities of frame {frame_idx+1}/{frames}")
    print(all_roi_intensities.shape)
    
    np.savetxt('array_contents.txt', all_roi_intensities, fmt='%f')
    # Calculate F0 for each ROI
    print("Calculating F0 values...")
    f0_per_roi = calculate_f0_per_roi_optimized(all_roi_intensities, frame_range, neighbors_list)
    
    # Create output array, optionally memory-mapped
    if use_memmap:
        dff_result, memmap_filename = create_memmap_array((frames, height, width))
    else:
        dff_result = np.zeros((frames, height, width), dtype=np.float32)
    
    # Process all frames for dF/F calculation
    print("Calculating dF/F for all frames...")
    for frame_idx in range(frames):
        dff_result[frame_idx] = dff_pixel_optimized(image_array[frame_idx], all_masks, f0_per_roi)
        
        if frame_idx % 10 == 0 or frame_idx == frames-1:
            print(f"Processed pixels of frame {frame_idx+1}/{frames}")
    
    return dff_result

if __name__ == "__main__":
    # Example usage
    time_s = time.time()
    print("Starting processing")
    
    #img_test = r"c:\Users\vixex\Desktop\b_lab_2025\paper_segmentation\celulas_modelo\model_cell\preprocessing\3uM_ROTE_1uM_FCCP_2.tif"
    #img = tifffile.imread(img_test)

    # Toggle memory mapping as needed
    use_memmap = False  # Set to True to use memory mapping
    
    dff_result = process_frame_frame_optimized(img_roi_perfect, roi_size, frame_range, use_memmap)
    
    time_f = time.time()
    print(f"Finished in {time_f - time_s:.2f} seconds")