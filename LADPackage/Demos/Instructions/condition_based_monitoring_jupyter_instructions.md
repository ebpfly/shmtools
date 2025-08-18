# Demo: Power Spectral Density Plotting for Condition Based Monitoring Data

**Dataset**: Accelerometer measurements from rotating machinery fault simulator, see ExampleUsages.pdf documentation

**Goal**: Plot power spectral density estimates for channel 3 of all instances in dataset

## Sequence

### Step 1: Import Condition Based Monitoring Data (SHMTools/Examples/ExampleData)
No inputs  
**Output**: Dataset = (TIME, CHANNELS, INSTANCES)  
**Output**: Damage States = (INSTANCES, 1)  
**Output**: State List = (INSTANCES,1)  
**Output**: Sampling Frequency = (scalar)

### Step 2: Power Spectral Density via Welch's Method (SHMFunctions/Feature Extraction/Spectral Analysis)
**Input**: X = Dataset (Step 1)  
**Input**: Window Length = leave as default  
**Input**: Overlap Length = leave as default  
**Input**: FFT Bins = leave as default  
**Input**: Sampling Frequency = Sampling Frequency (Step 1)  
**Input**: Use One-Sided PSD = leave as default  
**Output**: PSD Matrix = (NFFT, CHANNELS, INSTANCES)  
**Output**: Frequency Vector = (NFFT, 1)  
**Output**: PSD Range = (string)

### Step 3: Plot Power Spectral Density (SHMFunctions/Feature Extraction/Spectral Analysis)
**Input**: PSD Matrix = PSD Matrix (Step 2)  
**Input**: Channel Index = 3  
**Input**: PSD Range = PSD Range (Step 2)  
**Input**: Frequency Vector = Frequency Vector (Step 2)  
**Input**: Use dB Magnitude = leave as default  
**Input**: Plot Average PSD = leave as default  
**Input**: Axes Handle = leave as default  
**Output**: Axes Handle = (handle)