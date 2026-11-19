R2R: RAG evaluation on Google Colab (one model size per person)
=================================================================

You were given ONE model size: 3B, 7B, 14B or 32B. Run only the notebook for that size.

What's in this folder
---------------------
  r2r_colab_bundle.zip              code + data the notebook needs (do NOT unzip it)
  notebooks/R2R_RAG_Colab_3B.ipynb
  notebooks/R2R_RAG_Colab_7B.ipynb
  notebooks/R2R_RAG_Colab_14B.ipynb
  notebooks/R2R_RAG_Colab_32B.ipynb

Steps
-----
1. In Google Drive, create a folder named  R2R  at the top level of "My Drive".
   Upload r2r_colab_bundle.zip into it. The path must be exactly:
       MyDrive/R2R/r2r_colab_bundle.zip

2. Go to https://colab.research.google.com, then File -> Upload notebook, and pick YOUR size's notebook from notebooks/.

3. Runtime -> Change runtime type -> pick a GPU:
       3B, 7B  : any GPU (T4 is fine)
       14B     : T4 (16 GB) or better
       32B     : A100 (or L4). A T4 is too small.
   Use a fresh runtime. Don't run another model in the same session first.

4. Runtime -> Run all. Allow Google Drive access when asked.
   Expected time: about 30-90 min, depending on the GPU and size. The notebook also builds the medical
   search index, which adds about 20-40 min. 32B also downloads about 65 GB of weights.
   If Colab disconnects, open the notebook again and Run all. Finished parts are saved in Drive and skipped.

5. When the last cell prints "wrote .../r2r_results_<SIZE>.zip", send that file back:
       MyDrive/R2R/r2r_results_<SIZE>.zip     (e.g. r2r_results_14B.zip)
   The second-to-last cell prints a small table (accuracy change vs the direct answer). A screenshot of it
   is a useful extra.

If something goes wrong
-----------------------
* "needs about N GB of free GPU memory": the GPU is too small, or another model is still loaded.
  Pick a bigger GPU, or Runtime -> Restart session, then Run all.
* "No GPU": Runtime -> Change runtime type -> choose a GPU.
* "FileNotFoundError ... r2r_colab_bundle.zip": check the path in step 1 (folder name R2R, exact file name).
* Any other error: send a screenshot of the error together with your size.

Don't change the code or the settings cell, so that every size is run the same way.
