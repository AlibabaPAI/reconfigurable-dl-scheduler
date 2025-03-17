# Purpose of the folder transformers. 
The purpose of this folder is to enable real-time recording of the iteration time during the model training, as well as to quickly resume (by recording the number of steps completed) when the scheduler scales up or down the container resources. To guarantee that Morphling works properly, please perform the following steps before scheduling the workload:
1. Find the site-packages directory of the **transformers** library.
2. Replace the files in the library directory with the files from this folder.