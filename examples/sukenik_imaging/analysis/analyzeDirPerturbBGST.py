import argparse
import os
import imagej
import xarray as xr
import numpy as np
import glob
import pandas as pd
import skimage as ski
import matplotlib.pyplot as plt
from cellpose import models, core, io, plot
import tqdm
from pystackreg import StackReg
from skimage.transform import AffineTransform
from skimage.segmentation import relabel_sequential

model = models.CellposeModel(gpu=True)
sr = StackReg(StackReg.RIGID_BODY)

def segmentDAPI(img):
    '''DAPI segmentation using Otsu's method'''
    thresh = ski.filters.threshold_otsu(img)
    return img > thresh

def segmentFRET(img):
    flow_threshold = 0.4
    cellprob_threshold = 0.0
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
    

def main(dir):
    '''Main function to analyze images in the specified directory'''
    try:
        seqs = pd.read_csv(os.path.join(dir, 'seqs.csv'), index_col=0)
    except FileNotFoundError:
        print(f"Error: seqs.csv not found in {dir}. Please provide a valid directory.")
        return
    
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
    total_iterations = len(wells) * len(tiles)
    pbar = tqdm.tqdm(total=total_iterations, desc="Processing wells/tiles",ncols=150)
    for well in wells:
        construct=seqs.loc[well,'construct']
        print('on well %s construct %s'%(well, construct))
        for tile in tiles:
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
                    ## FRET masks Before / after - remove masks that touch edge and relabel to be sequential
                    maskFRET = segmentFRET(image[:,:,0])
                    edgeMasks = np.unique(np.concatenate([
                        maskFRET[0, :], maskFRET[-1, :], maskFRET[:, 0], maskFRET[:, -1]
                    ]))
                    maskFRET[np.isin(maskFRET, edgeMasks)] = 0
                    maskFRET, _, _ = relabel_sequential(maskFRET)

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

            
            # if mask is empty, continue
            if np.max(maskFRET) == 0:
                print(f"Warning: No cells detected in well {well}, tile {tile}. Skipping...")
                pbar.update(1)
                continue  # Skip to next tile


            maskFRETo = maskFRET.copy()

            ## ---------------------- BackgroundSubtraction ----------------------

            image[:,:,0], maskFRET = backgroundSubtract(image[:,:,0], maskFRET, 102, 24)
            image[:,:,1], maskFRET = backgroundSubtract(image[:,:,1], maskFRET, 105, 24)
            image[:,:,2], maskFRET = backgroundSubtract(image[:,:,2], maskFRET, 101, 20)

            image[:,:,4], maskFRET = backgroundSubtract(image[:,:,4], maskFRET, 102, 24)
            image[:,:,5], maskFRET = backgroundSubtract(image[:,:,5], maskFRET, 105, 24)
            image[:,:,6], maskFRET = backgroundSubtract(image[:,:,6], maskFRET, 101, 20)

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


            cells = np.arange(1, np.max(maskFRET)+1)
            for cell in cells:
                sel_cell_original = maskFRETo == cell
                sel_cell = maskFRET == cell
                sel_nuc = maskDAPI == cell
                sel_cyto = np.bitwise_xor(sel_cell, sel_nuc)

                if not np.any(sel_nuc) or not np.any(sel_cyto):
                    continue

                area_cell_original = np.sum(sel_cell_original)
                area_cell = np.sum(sel_cell)
                area_cyto = np.sum(sel_cyto)
                area_nuc = np.sum(sel_nuc)
                D_nuc = np.mean(image[:,:,0][sel_nuc])
                D_nuc_after = np.mean(image[:,:,4][sel_nuc])
                A_nuc = np.mean(image[:,:,1][sel_nuc])
                A_nuc_after = np.mean(image[:,:,5][sel_nuc])
                directA_nuc = np.mean(image[:,:,2][sel_nuc])
                directA_nuc_after = np.mean(image[:,:,6][sel_nuc])
                D_nuc_max = np.max(image[:,:,0][sel_nuc])
                D_nuc_max_after = np.max(image[:,:,4][sel_nuc])
                A_nuc_max = np.max(image[:,:,1][sel_nuc])
                A_nuc_max_after = np.max(image[:,:,5][sel_nuc])
                directA_nuc_max = np.max(image[:,:,2][sel_nuc])
                directA_nuc_max_after = np.max(image[:,:,6][sel_nuc])
                D_cyto = np.mean(image[:,:,0][sel_cyto])
                D_cyto_after = np.mean(image[:,:,4][sel_cyto])
                A_cyto = np.mean(image[:,:,1][sel_cyto])
                A_cyto_after = np.mean(image[:,:,5][sel_cyto])
                directA_cyto = np.mean(image[:,:,2][sel_cyto])
                directA_cyto_after = np.mean(image[:,:,6][sel_cyto])
                D_cyto_max = np.max(image[:,:,0][sel_cyto])
                D_cyto_max_after = np.max(image[:,:,4][sel_cyto])
                A_cyto_max = np.max(image[:,:,1][sel_cyto])
                A_cyto_max_after = np.max(image[:,:,5][sel_cyto])
                directA_cyto_max = np.max(image[:,:,2][sel_cyto])
                directA_cyto_max_after = np.max(image[:,:,6][sel_cyto])
                A_cyto_corr = A_cyto - 0.070 * directA_cyto - 0.4163 * D_cyto
                A_cyto_corr_after = A_cyto_after - 0.070 * directA_cyto_after - 0.4163 * D_cyto_after
                A_nuc_corr = A_nuc - 0.070 * directA_nuc - 0.4163 * D_nuc
                A_nuc_corr_after = A_nuc_after - 0.070 * directA_nuc_after - 0.4163 * D_nuc_after
                Ef_cyto = A_cyto_corr / (A_cyto_corr + D_cyto)
                Ef_cyto_after = A_cyto_corr_after / (A_cyto_corr_after + D_cyto_after)
                Ef_cyto_delta = Ef_cyto_after - Ef_cyto
                Ef_nuc = A_nuc_corr / (A_nuc_corr + D_nuc)
                Ef_nuc_after = A_nuc_corr_after / (A_nuc_corr_after + D_nuc_after)
                Ef_nuc_delta = Ef_nuc_after - Ef_nuc
                D_cell = np.mean(image[:,:,0][sel_cell])
                D_cell_after = np.mean(image[:,:,4][sel_cell])
                A_cell = np.mean(image[:,:,1][sel_cell])
                A_cell_after = np.mean(image[:,:,5][sel_cell])
                directA_cell = np.mean(image[:,:,2][sel_cell])
                directA_cell_after = np.mean(image[:,:,6][sel_cell])
                D_cell_max = np.max(image[:,:,0][sel_cell])
                D_cell_max_after = np.max(image[:,:,4][sel_cell])
                A_cell_max = np.max(image[:,:,1][sel_cell])
                A_cell_max_after = np.max(image[:,:,5][sel_cell])
                directA_cell_max = np.max(image[:,:,2][sel_cell])
                directA_cell_max_after = np.max(image[:,:,6][sel_cell])
                A_cell_corr = A_cell - 0.070 * directA_cell - 0.4163 * D_cell
                A_cell_corr_after = A_cell_after - 0.070 * directA_cell_after - 0.4163 * D_cell_after
                Ef_cell = A_cell_corr / (A_cell_corr + D_cell)
                Ef_cell_after = A_cell_corr_after / (A_cell_corr_after + D_cell_after)
                Ef_cell_delta = Ef_cell_after - Ef_cell
                perimeter_cell_original = ski.measure.perimeter(sel_cell_original)
                if perimeter_cell_original > 0 :
                    circ_cell_original = 4 * np.pi * area_cell_original / (perimeter_cell_original ** 2)
                else:
                    circ_cell_original = np.nan

                # Calculate quantile means for cytoplasm and nucleus
                # Direct Acceptor channel (image[:,:,2])
                cyto_intensities = image[:,:,2][sel_cyto]
                nuc_intensities = image[:,:,2][sel_nuc]
                
                # Calculate 10 quantiles
                cyto_quantiles = np.percentile(cyto_intensities, np.linspace(0, 100, 11))
                nuc_quantiles = np.percentile(nuc_intensities, np.linspace(0, 100, 11))
                
                # Calculate mean within each quantile bin for cytoplasm
                Q_cyto = {}
                for i in range(10):
                    mask_q = (cyto_intensities >= cyto_quantiles[i]) & (cyto_intensities < cyto_quantiles[i+1])
                    if i == 9:  # Include upper bound for last quantile
                        mask_q = (cyto_intensities >= cyto_quantiles[i]) & (cyto_intensities <= cyto_quantiles[i+1])
                    Q_cyto[f'directA_Q{i+1}_cyto'] = np.mean(cyto_intensities[mask_q]) if np.any(mask_q) else np.nan
                
                # Calculate mean within each quantile bin for nucleus
                Q_nuc = {}
                for i in range(10):
                    mask_q = (nuc_intensities >= nuc_quantiles[i]) & (nuc_intensities < nuc_quantiles[i+1])
                    if i == 9:  # Include upper bound for last quantile
                        mask_q = (nuc_intensities >= nuc_quantiles[i]) & (nuc_intensities <= nuc_quantiles[i+1])
                    Q_nuc[f'directA_Q{i+1}_nuc'] = np.mean(nuc_intensities[mask_q]) if np.any(mask_q) else np.nan

                # Calculate quantile means for cytoplasm and nucleus AFTER
                # Direct Acceptor channel (image[:,:,6])
                cyto_intensities = image[:,:,6][sel_cyto]
                nuc_intensities = image[:,:,6][sel_nuc]
                
                # Calculate 10 quantiles for cytoplasm
                cyto_quantiles = np.percentile(cyto_intensities, np.linspace(0, 100, 11))
                nuc_quantiles = np.percentile(nuc_intensities, np.linspace(0, 100, 11))
                
                # Calculate mean within each quantile bin for cytoplasm
                Q_cyto_after = {}
                for i in range(10):
                    mask_q = (cyto_intensities >= cyto_quantiles[i]) & (cyto_intensities < cyto_quantiles[i+1])
                    if i == 9:  # Include upper bound for last quantile
                        mask_q = (cyto_intensities >= cyto_quantiles[i]) & (cyto_intensities <= cyto_quantiles[i+1])
                    Q_cyto_after[f'directA_Q{i+1}_cyto_after'] = np.mean(cyto_intensities[mask_q]) if np.any(mask_q) else np.nan
                
                # Calculate mean within each quantile bin for nucleus
                Q_nuc_after = {}
                for i in range(10):
                    mask_q = (nuc_intensities >= nuc_quantiles[i]) & (nuc_intensities < nuc_quantiles[i+1])
                    if i == 9:  # Include upper bound for last quantile
                        mask_q = (nuc_intensities >= nuc_quantiles[i]) & (nuc_intensities <= nuc_quantiles[i+1])
                    Q_nuc_after[f'directA_Q{i+1}_nuc_after'] = np.mean(nuc_intensities[mask_q]) if np.any(mask_q) else np.nan
                

                df = pd.DataFrame({
                    'construct': construct,
                    'well': well,
                    'tile': tile,
                    'cell_idx': cell,
                    'area_cell_original': area_cell_original,
                    'area_cell': area_cell,
                    'area_cyto': area_cyto,
                    'area_nuc': area_nuc,
                    'D_cell': D_cell,
                    'D_cell_after': D_cell_after,
                    'D_cyto': D_cyto,
                    'D_cyto_after': D_cyto_after,
                    'D_nuc': D_nuc,
                    'D_nuc_after': D_nuc_after,
                    'A_cell': A_cell,
                    'A_cell_after': A_cell_after,
                    'A_cyto': A_cyto,
                    'A_cyto_after': A_cyto_after,
                    'A_nuc': A_nuc,
                    'A_nuc_after': A_nuc_after,
                    'A_cell_corr': A_cell_corr,
                    'A_cell_corr_after': A_cell_corr_after,
                    'A_cyto_corr': A_cyto_corr,
                    'A_cyto_corr_after': A_cyto_corr_after,
                    'A_nuc_corr': A_nuc_corr,
                    'A_nuc_corr_after': A_nuc_corr_after,
                    'Ef_cell': Ef_cell,
                    'Ef_cell_after': Ef_cell_after,
                    'Ef_cell_delta': Ef_cell_delta,
                    'Ef_cyto': Ef_cyto,
                    'Ef_cyto_after': Ef_cyto_after,
                    'Ef_cyto_delta': Ef_cyto_delta,
                    'Ef_nuc': Ef_nuc,
                    'Ef_nuc_after': Ef_nuc_after,
                    'Ef_nuc_delta': Ef_nuc_delta,
                    'directA_cell': directA_cell,
                    'directA_cell_after': directA_cell_after,
                    'directA_cyto': directA_cyto,
                    'directA_cyto_after': directA_cyto_after,
                    'directA_nuc': directA_nuc,
                    'directA_nuc_after': directA_nuc_after,
                    'D_cell_max': D_cell_max,
                    'D_cell_max_after': D_cell_max_after,
                    'D_cyto_max': D_cyto_max,
                    'D_cyto_max_after': D_cyto_max_after,
                    'D_nuc_max': D_nuc_max,
                    'D_nuc_max_after': D_nuc_max_after,
                    'A_cell_max': A_cell_max,
                    'A_cell_max_after': A_cell_max_after,
                    'A_cyto_max': A_cyto_max,
                    'A_cyto_max_after': A_cyto_max_after,
                    'A_nuc_max': A_nuc_max,
                    'A_nuc_max_after': A_nuc_max_after,
                    'directA_cell_max': directA_cell_max,
                    'directA_cell_max_after': directA_cell_max_after,
                    'circ_cell_original' : circ_cell_original,
                    **Q_cyto,
                    **Q_nuc,
                    **Q_cyto_after,
                    **Q_nuc_after,
                }, index=[0])
                cells_df = pd.concat([cells_df, df])
            pbar.update(1)
    out_csv = os.path.join(dir, 'cells_df_BGSTperturbq.csv')
    cells_df.to_csv(out_csv)
    print(f"Saved output to {out_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze cells and output CSV to directory.")
    parser.add_argument("directory", help="Directory containing image files and seqs.csv")
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
    main(args.directory)