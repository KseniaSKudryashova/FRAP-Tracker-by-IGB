🧪 FRAP-Tracker by IGB

A Python-based software suite for processing FRAP images from Leica Stellaris 5 microscope for subsequent analysis in EasyFRAP

FRAP-Tracker is designed to analyze Fluorescence Recovery After Photobleaching experiments where cellular structures move during image acquisition. The program extracts intensity values from user-defined regions across entire time-lapse sequences, preparing data for EasyFRAP compatibility.

FRAP-Tracker BASIC handles experiments where only the nucleus moves while the bleached foci remains fixed relative to nuclear boundaries (e.g., histones, chromatin-bound proteins). The program tracks nucleus movement using phase correlation and calculates foci position as a fixed offset from the nucleus.

FRAP-Tracker ADCANCED addresses more complex scenarios where both the nucleus and the bleached foci move independently during acquisition (e.g., nuclear bodies like Cajal bodies, PML bodies). This version actively tracks both structures: nucleus via phase correlation and foci by re-detecting intensity maxima in each frame within a search radius.
