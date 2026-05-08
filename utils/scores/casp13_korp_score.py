import pandas as pd
import os.path as osp
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

class CASP13KorpScores:
    def __init__(self, folder_name: str) -> None:
        # eg.: ./exp_pp_018_run_2024-05-21_163016__933628/exp_pp_018_test_on_CASP13_korp_test_2024-06-04_181328
        self.folder_name = folder_name

    def run(self):
        df1=pd.read_csv('/scratch/sx801/scripts/korp_assessment_scripts/Korp6Dv1/CASP13_split.csv')

        res_csv = glob(osp.join(self.folder_name, "pp*.af22_50.csv"))
        assert len(res_csv) == 1, f"Zero or multiple result csv found: {res_csv}"
        df2 = pd.read_csv(res_csv[0])

        # Function to extract and clean the filename
        def clean_filename(path):
            filename = path.split('/')[-1]  # Split by '/' and take the last part
            return filename.replace('_protein', '')  # Remove the '_protein' part

        # Apply the function to the 'protein_file' column and store results in a new column
        df2['PDB_file'] = df2['protein_file'].apply(clean_filename)

        # Function to extract the standardized part of the filename
        def standardize_filename(filename):
            # Use regex to remove unwanted parts after the core identifier
            return pd.Series(filename).replace(r'(_s\d+m\d+)?\.pdb', '', regex=True).iloc[0]

        # Apply the function to the 'PDB_file' column and store results in a new column
        df2['case_name'] = df2['PDB_file'].apply(standardize_filename)

        df=pd.merge(df1,df2,on='PDB_file')
        df_work=df[['PDB_file','Tmscore','GDTTS','score_normalized','case_name']]

        # ----------------------------N_N-------------------------------- #
        # 1: N. Number of benchmark cases where the native conformation (NN) or 
        # the best decoy (ND) has the lowest energy.
        # Best Decoy: with the highest GDT-TS(for CASP13, still use GDT-TS to 100% reproduce results)

        # Assuming df_work is your DataFrame loaded with the correct columns

        # Step 1: Use ascending=True to make the model with the lowest score_normalized as the top-1
        df_work['rank_by_score'] = df_work.groupby('case_name')['score_normalized'].rank(ascending=False, method='first')

        # Identify the top-1 ranked model for each case_name
        top_1_models = df_work[df_work['rank_by_score'] == 1]

        # Find the highest GDTTS among decoys (excluding natives) for each case_name
        decoys_df = df_work[df_work['GDTTS'] < 1.0]  # Ensure only decoys are considered
        highest_GDTTS_decoys = decoys_df.loc[decoys_df.groupby('case_name')['GDTTS'].idxmax()]

        # Check if the top-1 ranked model has the highest GDTTS among decoys
        top_1_models = top_1_models.merge(highest_GDTTS_decoys[['case_name', 'GDTTS']], on='case_name', suffixes=('', '_highest_decoy'))
        top_1_models['is_highest_GDTTS_among_decoys'] = top_1_models['GDTTS'] == top_1_models['GDTTS_highest_decoy']

        # Count how many top-1 ranked models are native
        top_1_models['is_native'] = top_1_models['GDTTS'] == 1.0
        native_count = top_1_models['is_native'].sum()

        # Count how many top-1 ranked models have the highest GDTTS among decoys
        highest_GDTTS_count = top_1_models['is_highest_GDTTS_among_decoys'].sum()

        # Output the results
        print(f"Count of top-1 ranked models that are native structures: {native_count}")
        print(f"Count of top-1 ranked models that have the highest GDTTS among decoys: {highest_GDTTS_count}")

        # Step 3: Count how many top-1 ranked models are either native or have the highest GDTTS among decoys
        valid_top_1_count = top_1_models[(top_1_models['is_native'] | top_1_models['is_highest_GDTTS_among_decoys'])].shape[0]

        print(f"Count of top-1 ranked models that are either native structures or have the highest GDTTS among decoys: {valid_top_1_count}")


        N_N=native_count

        # ----------------------------N_D-------------------------------- #
        # Assuming df_work is your DataFrame loaded with the correct columns

        # Step 1: Use ascending=True to make the model with the lowest score_normalized as the top-1
        df_work2=df_work
        df_work=df_work[df_work['GDTTS'] < 1.0] 
        df_work['rank_by_score'] = df_work.groupby('case_name')['score_normalized'].rank(ascending=False, method='first')

        # Identify the top-1 ranked model for each case_name
        top_1_models = df_work[df_work['rank_by_score'] == 1]

        # Find the highest GDTTS among decoys (excluding natives) for each case_name
        decoys_df = df_work[df_work['GDTTS'] < 1.0]  # Ensure only decoys are considered
        highest_GDTTS_decoys = decoys_df.loc[decoys_df.groupby('case_name')['GDTTS'].idxmax()]

        # Check if the top-1 ranked model has the highest GDTTS among decoys
        top_1_models = top_1_models.merge(highest_GDTTS_decoys[['case_name', 'GDTTS']], on='case_name', suffixes=('', '_highest_decoy'))
        top_1_models['is_highest_GDTTS_among_decoys'] = top_1_models['GDTTS'] == top_1_models['GDTTS_highest_decoy']

        # Count how many top-1 ranked models are native
        top_1_models['is_native'] = top_1_models['GDTTS'] == 1.0
        native_count = top_1_models['is_native'].sum()

        # Count how many top-1 ranked models have the highest GDTTS among decoys
        highest_GDTTS_count = top_1_models['is_highest_GDTTS_among_decoys'].sum()

        # Output the results
        print(f"Count of top-1 ranked models that are native structures: {native_count}")
        print(f"Count of top-1 ranked models that have the highest GDTTS among decoys: {highest_GDTTS_count}")

        # Step 3: Count how many top-1 ranked models are either native or have the highest GDTTS among decoys
        valid_top_1_count = top_1_models[(top_1_models['is_native'] | top_1_models['is_highest_GDTTS_among_decoys'])].shape[0]

        print(f"Count of top-1 ranked models that are either native structures or have the highest GDTTS among decoys: {valid_top_1_count}")
        df_work=df_work2


        N_D=highest_GDTTS_count

        # ------------------------------Z_N and Z_D------------------------------ #
        # 2: Z. This metric measures the number of standard deviations (σ) between the energy of the native structure 
        # (Enative) and the mean energy of the decoys (μ): ZN = (μ − Enative)/σ. 
        # In the same way, it can be defined with respect to the best decoy (closest to the native) as: 
        # ZD = (μ − Ebest_decoy)/σ.

        # Assuming df_work is your DataFrame
        df_work['type'] = np.where(df_work['Tmscore'] == 1, 'native', 'decoy')

        # Group by case_name to handle multiple cases independently
        grouped = df_work.groupby('case_name')

        results = []

        for name, group in grouped:
            # Extract energies for decoys and the native structure if exists
            decoy_energies = group[group['type'] == 'decoy']
            native_energy = group[group['type'] == 'native']['score_normalized'].values

            # Only compute Z-scores if there's at least one native and decoys
            if len(native_energy) > 0 and not decoy_energies.empty:
                native_energy = native_energy[0]  # Assuming there's only one native per case
                mu = decoy_energies['score_normalized'].mean()
                sigma = decoy_energies['score_normalized'].std()

                # Find the best decoy based on the highest Tmscore
        #         best_decoy = decoy_energies[decoy_energies['Tmscore'] == decoy_energies['Tmscore'].max()]
                best_decoy = decoy_energies[decoy_energies['GDTTS'] == decoy_energies['GDTTS'].max()]
                if not best_decoy.empty:
                    best_decoy_energy = best_decoy['score_normalized'].values[0]

                    # Calculate ZN
                    ZN = (mu - native_energy) / sigma

                    # Calculate ZD
                    ZD = (mu - best_decoy_energy) / sigma

                    results.append({
                        'case_name': name,
                        'ZN': ZN,
                        'ZD': ZD
                    })

        # Convert results to DataFrame for better visualization
        results_df = pd.DataFrame(results)
        Z_N=round(-results_df['ZN'].mean(),2)
        Z_D=round(-results_df['ZD'].mean(),2)

        # ------------------------------Loss------------------------------ #
        # 3: Loss. Loss of quality between the best decoy available and the 
        # lowest energy model in percentage of GDT_TS score.
        # Assuming df_work is your DataFrame

        # Step 1: Exclude native structures from the analysis
        # Assuming native structures have TMscore == 1.0
        decoys_df = df_work[df_work['Tmscore'] < 1.0]

        # Step 2: Find the decoy with the highest GDTTS for each case_name
        best_decoy_indices = decoys_df.groupby('case_name')['GDTTS'].idxmax()
        best_decoys = decoys_df.loc[best_decoy_indices]

        # Step 3: Find the decoy with the lowest score_normalized for each case_name (highest score indicates lowest energy)
        lowest_energy_indices = decoys_df.groupby('case_name')['score_normalized'].idxmax()
        lowest_energy_models = decoys_df.loc[lowest_energy_indices]

        # Step 4: Ensure the data is aligned correctly by setting a common index (case_name) for comparison
        best_decoys.set_index('case_name', inplace=True)
        lowest_energy_models.set_index('case_name', inplace=True)

        # Step 5: Calculate the loss of quality
        loss_of_quality =  ( best_decoys['GDTTS']-lowest_energy_models['GDTTS']) * 100

        # Result: Reset index for display
        loss_of_quality_results = loss_of_quality.reset_index()
        print(loss_of_quality_results)

        # Step 6: Calculate the average loss of quality
        average_loss_of_quality = loss_of_quality.mean()

        # Print the average loss of quality
        print(f"Average Loss of Quality: {average_loss_of_quality}%")
        loss=round(average_loss_of_quality,1)

        # ------------------------------N0.5------------------------------ #
        # 4. N0.5. Number of cases in the benchmark in where the lowest energy 
        # decoy has a GDT_TS score larger than 0.5.

        # Assuming df_work is your DataFrame
        # Exclude native structures (assuming TMscore < 1.0 means non-native)
        decoys_df = df_work[df_work['Tmscore'] < 1.0]

        # Find the index of the decoy with the highest score_normalized for each case_name
        # This is considered as the decoy with the lowest energy due to high score_normalized being equivalent to low energy
        lowest_energy_indices = decoys_df.groupby('case_name')['score_normalized'].idxmax()
        lowest_energy_decoys = decoys_df.loc[lowest_energy_indices]

        # Filter to keep only those where GDTTS > 0.5
        valid_cases = lowest_energy_decoys[lowest_energy_decoys['GDTTS'] > 0.5]

        # Count these cases
        n0_5_count = valid_cases.shape[0]

        # Output the result
        print(f"Number of cases where the decoy with the lowest energy (highest score_normalized) has a GDT_TS score greater than 0.5: {n0_5_count}")

        N0_5=n0_5_count

        # ------------------------------Rank------------------------------ #
        # 5. Rank. Rank of the lowest energy decoy. To facilitate comparison between benchmarks, 
        # it is expressed as the percentage of the total number of decoys for each target.
        # Assuming df_work is your DataFrame

        # Exclude native structures if necessary (assuming TMscore of 1.0 denotes native)
        decoys_df = df_work[df_work['Tmscore'] < 1.0]

        # Step 1: Find the decoy with the highest GDTTS for each case_name
        best_decoy_indices = decoys_df.groupby('case_name')['GDTTS'].idxmax()
        best_decoys = decoys_df.loc[best_decoy_indices]

        # Step 2: Rank all decoys by score_normalized within each case_name
        # Here, we make sure to rank in ascending order since higher score_normalized indicates lower energy
        decoys_df['rank_by_score'] = decoys_df.groupby('case_name')['score_normalized'].rank(ascending=False, method='first') - 1  # Subtract 1 here to start ranking from 0

        # Step 3: Get the rank_by_score for the best GDTTS decoys
        best_decoys['energy_rank'] = best_decoys.apply(lambda x: decoys_df.loc[x.name, 'rank_by_score'], axis=1)

        # Step 4: Calculate the total number of decoys for each case_name
        best_decoys['total_decoys'] = best_decoys['case_name'].apply(lambda x: decoys_df[decoys_df['case_name'] == x].shape[0])

        # Step 5: Calculate the rank percentage for the best GDTTS decoys based on score_normalized
        best_decoys['rank_percentage'] = (best_decoys['energy_rank'] / (best_decoys['total_decoys'] - 1)) * 100  # Adjust here to factor in 0-based indexing

        # Print the results
        print(best_decoys[['case_name', 'PDB_file', 'GDTTS', 'score_normalized', 'energy_rank', 'total_decoys', 'rank_percentage']])
        rank=round(best_decoys['rank_percentage'].mean(),1)

         # ------------------------------ΔAUC------------------------------ #
         # 6 ΔAUC. Is the difference between two areas under curves. 
         # The first curve is the number of targets with at least one decoy over a given value of GDT_TS, 
         # and the second is the number of targets in where the lowest energy decoy scores better 
         # than a given GDT_TS value (see Supplementary Fig. S1). 
         # To focus only on cases where at least a good model is present, 
         # this difference was restricted between 0.5 and 1.0 GDT_TS scores. 
         # As before, it is expressed as a percentage. Low percentages indicate that the lowest 
         # energy decoys are very close to the best available and vice versa.
         # Assuming df_work is your DataFrame with columns 'case_name', 'GDTTS', and 'score_normalized'

        # Step 1: Filter the data
        # filtered_df = df_work[(df_work['GDTTS'] >= 0) & (df_work['GDTTS'] < 1.0)]
        filtered_df = df_work[(df_work['GDTTS'] < 1.0)]
        # Step 2: Initialize lists to store counts for each threshold
        targets_with_decoy_above_threshold = []
        targets_with_lowest_energy_above_threshold = []

        # Step 3: Iterate over threshold values
        thresholds = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        thresholds = np.arange(0.5, 1.001, 0.001)
        for threshold in thresholds:
            # Count cases where at least one decoy has GDT_TS >= threshold
            count = ((filtered_df['GDTTS'] >= threshold).groupby(filtered_df['case_name']).any()).sum()
            targets_with_decoy_above_threshold.append(count)
            
            # Count cases where lowest energy decoy has GDT_TS >= threshold
            lowest_energy_indices = filtered_df.groupby('case_name')['score_normalized'].idxmax()
            lowest_energy_models = filtered_df.loc[lowest_energy_indices]
            targets_with_lowest_energy_above_threshold.append((lowest_energy_models['GDTTS'] >= threshold).sum())

        # Step 4: Calculate the area under the two curves
        area_decoy_above = np.trapz(targets_with_decoy_above_threshold, thresholds)


        area_lowest_energy_above = np.trapz(targets_with_lowest_energy_above_threshold, thresholds)



        # Step 5: Calculate ΔAUC
        delta_auc_percentage = (area_decoy_above-area_lowest_energy_above) 

        # Step 6: Visualize the results
        plt.plot(thresholds, targets_with_decoy_above_threshold, label='At least one decoy')
        plt.plot(thresholds, targets_with_lowest_energy_above_threshold, label='Lowest energy decoy')
        plt.xlabel('Threshold GDT_TS')
        plt.ylabel('Number of targets')
        plt.title('ΔAUC Curve')
        plt.legend()
        plt.grid(True)
        plt.savefig(osp.join(self.folder_name, "casp13_delta_auc.png"))
        plt.close()

        print(round(delta_auc_percentage*100/area_decoy_above,1))
        delta_auc=round(delta_auc_percentage*100/area_decoy_above,1)

        # Assuming df_work is your DataFrame with necessary columns and case groupings
        # Filter out cases where GDT-TS is not less than 1.0
        filtered_df = df_work[df_work['GDTTS'] < 1.0]

        # Drop any rows with missing values in either 'GDTTS' or 'score_normalized'
        df_cleaned = filtered_df.dropna(subset=['GDTTS', 'score_normalized'])

        # Group by case and calculate correlation within each group
        results = df_cleaned.groupby('case_name').apply(
            lambda group: pearsonr(group['GDTTS'], group['score_normalized'])[0]
        )

        # Filter out cases where correlation couldn't be calculated (e.g., due to constant values)
        valid_results = results.dropna()

        # Calculate the average of the correlation coefficients
        average_r = valid_results.mean()

        print(f"Average Pearson's correlation coefficient (r): {average_r}")

        pear_cor=round(average_r,2)

        fin_res = {"N_N": N_N, "Z_N": Z_N, "N_D": N_D, "Z_D": Z_D, "LOSS": loss, "N0_5": N0_5,
                   "RANK": rank, "DELTA_AUC": delta_auc, "PEAR_COR": pear_cor}
        fin_df = pd.DataFrame(fin_res, index=[0])
        fin_df.to_csv(osp.join(self.folder_name, "casp13_res.csv"), index=False)
        fin_df.to_excel(osp.join(self.folder_name, "casp13_res.xlsx"), index=False)
