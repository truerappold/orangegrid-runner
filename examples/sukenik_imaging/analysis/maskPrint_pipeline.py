import argparse
import os
import imagej
import xarray as xr
import numpy as np
import glob
import pandas as pd
import skimage as ski
import matplotlib
import matplotlib.pyplot as plt
from cellpose import models, core, io, plot
import tqdm
from pystackreg import StackReg
from skimage.transform import AffineTransform
from skimage import segmentation
from skimage.segmentation import relabel_sequential
from scipy.sparse import csr_matrix
from sklearn.metrics import jaccard_score

matplotlib.use('Agg')

model = models.CellposeModel(gpu=True)
sr = StackReg(StackReg.RIGID_BODY)

def segmentDAPI(img):
    '''DAPI segmentation using Otsu's method'''
    thresh = ski.filters.threshold_otsu(img)
    return img > thresh

def segmentFRET(img):
    flow_threshold = 1
    cellprob_threshold = 0
    tile_norm_blocksize = 0
    io.logger_setup()
    masks, flows, styles = model.eval(img, batch_size=32, flow_threshold=flow_threshold, progress=True, cellprob_threshold=cellprob_threshold,
                                    normalize={"tile_norm_blocksize": tile_norm_blocksize})
    return(masks)

def backgroundSubtract(img, maskFRET, background, threshold):
    # Subtract flat background
    tmp = img.astype(np.int32) - background
    # Set any values less than 1 to 1
    tmp[tmp < 1] = 1
    # Create a copy to avoid modifying the original
    maskFRET_corrected = maskFRET.copy()
    
    # Check each mask in maskFRET and zero out regions around negative/zero values
    # Check each mask in maskFRET and zero out regions around negative/zero values
    for mask_id in np.unique(maskFRET_corrected):
        if mask_id == 0:  # Skip background
            continue
        # Get mask for this cell
        mask_region = maskFRET_corrected == mask_id
        # Find locations where pixels are <= threshold
        noise_locations = np.where((tmp <= threshold) & mask_region)
        
        # Create a binary mask of all noise locations at once
        noise_mask = np.zeros_like(maskFRET_corrected, dtype=bool)
        noise_mask[noise_locations] = True
        
        # Dilate the noise mask to create 5x5 squares (radius=2 gives 5x5)
        dilated_noise = ski.morphology.binary_dilation(noise_mask, footprint=ski.morphology.footprint_rectangle((5, 5)))

        
        # Zero out maskFRET at dilated locations
        maskFRET_corrected[dilated_noise] = 0
    
    # Return to uint16
    img_corrected = tmp.astype(np.uint16)
    
    return img_corrected, maskFRET_corrected
    

def main(dir, out_dir):
    '''Main function to analyze images in the specified directory'''
    try:
        seqs = pd.read_csv(os.path.join(dir, 'seqs.csv'), index_col=0)
    except FileNotFoundError:
        print(f"Error: seqs.csv not found in {dir}. Please provide a valid directory.")
        return
    
    # Create output directory if it doesn't exist
    os.makedirs(out_dir, exist_ok=True)
    
    filelist = glob.glob(os.path.join(dir+'Pics/'+'*.vsi'))

    if not filelist:
        print(f"No VSI files found in {dir}. Please provide a valid directory with image files.")
        return
    channels = ['FRET, FRET', 'mNG', 'DAPI']
    wells = list(set([os.path.basename(a).split('-')[0] for a in filelist]))
    original_wells = wells.copy()
    wells = [well for well in wells if well in seqs.index]

    if len(wells) < len(original_wells):
        missing_wells = [w for w in original_wells if w not in wells]
        # Create a txt file with missing wells instead of printing
        missing_wells_file = os.path.join(dir, 'missing_wells.txt')
        with open(missing_wells_file, 'w') as f:
            f.write("Wells found in images but not in seqs.csv:\n")
            for well in missing_wells:
                f.write(f"{well}\n")
        print(f"Missing wells list saved to {missing_wells_file}")

    if not wells:
        print(f"No matching wells found between image files and seqs.csv in {dir}.")
        return

    print(f"Found wells: {wells}")
    tiles = np.arange(1,10)
    cells_df = pd.DataFrame()
    total_iterations = len(wells) * len(tiles[8:9])
    pbar = tqdm.tqdm(total=total_iterations, desc="Processing wells/tiles",ncols=150)
    for well in wells:
        construct=seqs.loc[well,'construct']
        print('on well %s construct %s'%(well, construct))
        for tile in tiles[8:9]:
            print('on tile %i'%tile)
            image = np.zeros([2304,2304,8],dtype='uint16')
            tmatrix = None
            tmatrixP = None
            tmatrixW = None
            for channel in channels:
                filename = dir+'Pics/'+well+'-'+str(tile)+'_'+channel+'.vsi'
                filenameAfter = dir+'Pics/'+'1'+well+'-'+str(tile)+'_'+channel+'.vsi'
                dataset = ij.io().open(filename)
                datasetAfter = ij.io().open(filenameAfter)
                if channel=='FRET, FRET':
                    ## donor B/A
                    image[:,:,0]=ij.py.from_java(dataset[:,:,1])
                    image[:,:,4]=ij.py.from_java(datasetAfter[:,:,1])
                    ## register D-A to D-B
                    ref = image[:,:,0] >= ski.filters.threshold_otsu(image[:,:,0])
                    mov = image[:,:,4] >= ski.filters.threshold_otsu(image[:,:,4])
                    tmatrixP = sr.register(ref, mov)
                    tmatrixW = sr.register(ref, mov)
                    image[:,:,4] = ski.transform.warp(image[:,:,4], tmatrixP, preserve_range=True).astype('uint16')

                    ## acceptor B/A
                    image[:,:,1]=ij.py.from_java(dataset[:,:,0])
                    image[:,:,5]=ij.py.from_java(datasetAfter[:,:,0])
                    ## get tmatrix and register A-B to D-B
                    ref = image[:,:,0] >= ski.filters.threshold_otsu(image[:,:,0])
                    mov = image[:,:,1] >= ski.filters.threshold_otsu(image[:,:,1])
                    tmatrix = sr.register(ref, mov)
                    image[:,:,1] = ski.transform.warp(image[:,:,1], tmatrix, preserve_range=True).astype('uint16')
                    ## register A-A to D-A
                    ref = image[:,:,4] >= ski.filters.threshold_otsu(image[:,:,4])
                    mov = image[:,:,5] >= ski.filters.threshold_otsu(image[:,:,5])
                    tmatrixP = sr.register(ref, mov)
                    image[:,:,5] = ski.transform.warp(image[:,:,5], tmatrixP, preserve_range=True).astype('uint16')
                if channel=='mNG':
                    ## register to FRET donor using tmatrix from acceptor
                    image[:,:,2] = ij.py.from_java(dataset)
                    image[:,:,2] = ski.transform.warp(image[:,:,2], tmatrix, preserve_range=True).astype('uint16')
                    ## register dA-A to D-A using tmatrixP from acceptor
                    image[:,:,6] = ij.py.from_java(datasetAfter)
                    image[:,:,6] = ski.transform.warp(image[:,:,6], tmatrixP, preserve_range=True).astype('uint16')
                if channel == 'DAPI':
                    ## 
                    image[:,:,3]=ij.py.from_java(dataset)
                    ## Transform DAPI after using tmatrix, to approximately match after - before
                    image[:,:,7]=ij.py.from_java(datasetAfter)
                    image[:,:,7] = ski.transform.warp(image[:,:,7], tmatrixW, preserve_range=True).astype('uint16')
                    ## DAPI mask
                    maskDAPI = segmentDAPI(image[:,:,3])
                    maskDAPIafter = segmentDAPI(image[:,:,7])

            ## FRET masks Before / after - remove masks that touch edge and relabel to be sequential
            maskFRET = segmentFRET(image[:,:,2])
            maskFRETo = maskFRET.copy()
            edgeMasks = np.unique(np.concatenate([
                maskFRET[0, :], maskFRET[-1, :], maskFRET[:, 0], maskFRET[:, -1]
            ]))
            maskFRET[np.isin(maskFRET, edgeMasks)] = 0
            maskFRET, _, _ = relabel_sequential(maskFRET)

            # Remove these lines that convert to binary:
            # maskFRET[maskFRET >= 1] = 1
            # maskFRET_labeled = ski.measure.label(maskFRET)
            # maskFRET, _, _ = relabel_sequential(maskFRET_labeled)
            
            # if mask is empty, continue
            if np.max(maskFRET) == 0:
                print(f"Warning: No cells detected in well {well}, tile {tile}. Skipping...")
                pbar.update(1)
                continue  # Skip to next tile

            ## ---------------------- BackgroundSubtraction ----------------------

            #image[:,:,0], maskFRET = backgroundSubtract(image[:,:,0], maskFRET, 102, 24)
            #image[:,:,1], maskFRET = backgroundSubtract(image[:,:,1], maskFRET, 105, 24)
            #image[:,:,2], maskFRET = backgroundSubtract(image[:,:,2], maskFRET, 101, 20)

            #image[:,:,4], maskFRET = backgroundSubtract(image[:,:,4], maskFRET, 102, 24)
            #image[:,:,5], maskFRET = backgroundSubtract(image[:,:,5], maskFRET, 105, 24)
            #image[:,:,6], maskFRET = backgroundSubtract(image[:,:,6], maskFRET, 101, 20)

            ## --------- Backgroumnd Subtraction Done ------

            maskDAPI = maskDAPI * maskFRET
            maskDAPIafter = maskDAPIafter * maskFRET

            # For each FRET cell, keep only the largest DAPI particle and fill holes
            for cell_id in np.unique(maskFRET):
                if cell_id == 0:  # Skip background
                    continue
                # Get DAPI mask for this cell
                cell_dapi = (maskDAPI == cell_id)
                cell_dapi_after = (maskDAPIafter == cell_id)
                if not np.any(cell_dapi) or not np.any(cell_dapi_after):
                    continue
                
                # Label connected components within this cell's DAPI mask
                labeled_dapi = ski.measure.label(cell_dapi)
                labeled_dapi_after = ski.measure.label(cell_dapi_after)

                # Find the largest component
                regions = ski.measure.regionprops(labeled_dapi)
                regions_after = ski.measure.regionprops(labeled_dapi_after)

                if len(regions_after) == 0:
                    continue

                largest_region = max(regions, key=lambda r: r.area)
                largest_region_after = max(regions_after, key=lambda r: r.area)

                # Keep only the largest component
                largest_mask = (labeled_dapi == largest_region.label)
                largest_mask_after = (labeled_dapi_after == largest_region_after.label)

                # Fill holes in the largest mask -- pay attention to area threshold!!!
                filled_mask = ski.morphology.remove_small_holes(largest_mask, area_threshold=200)
                filled_mask_after = ski.morphology.remove_small_holes(largest_mask_after, area_threshold=200)

                # Update maskDAPI: zero out this cell's region, then add back the filled largest component
                maskDAPI[maskFRET == cell_id] = 0
                maskDAPI[filled_mask] = cell_id
                maskDAPIafter[maskFRET == cell_id] = 0
                maskDAPIafter[filled_mask_after] = cell_id


# Create blank image with same shape as input
            blank_image = np.zeros_like(image[:, :, 0], dtype=np.uint8)
            
            # Create RGB image for colored masks
            colored_image = np.zeros((*maskFRET.shape, 3), dtype=np.float32)
            
            # Get number of cells for color gradient
            num_cells = np.max(maskFRET)
            cmap = plt.cm.get_cmap('rainbow', num_cells)
            
            # Fill each FRET mask with a gradient color
            for cell_id in np.unique(maskFRET):
                if cell_id == 0:  # Skip background
                    continue
                # Create binary mask for this cell
                cell_mask = (maskFRET == cell_id)
                # Get color from gradient
                color = cmap((cell_id - 1) / num_cells)[:3]  # Get RGB, skip alpha
                # Fill the colored image
                for i in range(3):
                    colored_image[:, :, i][cell_mask] = color[i]
            
            # Create high-resolution figure with black background
            fig_overlay, ax_overlay = plt.subplots(1, 1, figsize=(16, 16), facecolor='black')
            ax_overlay.set_facecolor('black')
            
            # Show colored masks
            ax_overlay.imshow(colored_image)
            
            # Add nucleus contours and cell numbers
            for cell_id in np.unique(maskFRET):
                if cell_id == 0:  # Skip background
                    continue
                
                # Find and plot nucleus contour in white
                nucleus_mask = (maskDAPI == cell_id)
                if np.any(nucleus_mask):
                    contours_nucleus = ski.measure.find_contours(nucleus_mask.astype(float), 0.5)
                    for contour in contours_nucleus:
                        ax_overlay.plot(contour[:, 1], contour[:, 0], 'w-', linewidth=1.5, alpha=1.0)
                
                # Add cell number at centroid
                cell_mask = (maskFRET == cell_id)
                props = ski.measure.regionprops(cell_mask.astype(int))
                if props:
                    centroid = props[0].centroid
                    ax_overlay.text(centroid[1], centroid[0], str(cell_id), 
                                  color='white', fontsize=8, ha='center', va='center',
                                  weight='bold')
            
            ax_overlay.axis('off')
            
            plt.tight_layout()
            out_overlay_path = os.path.join(out_dir, f"{well}_tile{tile}_mask_contour.tif")
            plt.savefig(out_overlay_path, dpi=300, bbox_inches='tight', pad_inches=0, facecolor='black')
            plt.close(fig_overlay)
            print(f"Saved mask contour to {out_overlay_path}")
            pbar.update(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create mask overlay contours and save as TIFF files.")
    parser.add_argument("directory", help="Directory containing image files and seqs.csv")
    parser.add_argument("output_dir", help="Output directory for saving mask contour TIFF files")
    args = parser.parse_args()
    io.logger_setup()
    try:
        model = models.CellposeModel(gpu=True)
    except:
        print("Error initializing Cellpose model. Ensure that Cellpose is installed and configured correctly.")
    if not model:
        print("Cellpose model could not be initialized. Exiting.")
    try:
        ij = imagej.init('sc.fiji:fiji:2.14.0')
    except Exception as e:
        print(f"Error initializing ImageJ: {e}")
    print(f"ImageJ version: {ij.getVersion()}")
    main(args.directory, args.output_dir)