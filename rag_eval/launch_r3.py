"""Spawn the R3 orchestrator on the DEPLOYED app, so it does not depend on a local client staying connected.

  modal deploy rag_eval/rag_eval_modal.py
  python3 rag_eval/launch_r3.py            # prints the FunctionCall id
  modal app logs r2r-rag-eval              # follow progress
"""
import modal

R3_OPTIONS = ["med_bm25_rr", "med_dense", "med_dense_rr", "med_hybrid_rr", "med_hybrid_opt_rr",
              "med_hybrid_rr__L3000", "medtb_dense_rr", "medtb_hybrid_rr"]

if __name__ == "__main__":
    f = modal.Function.from_name("r2r-rag-eval", "orchestrate_r3")
    call = f.spawn(["validation", "test"], ["3B", "7B", "14B", "32B"], R3_OPTIONS)
    print(call.object_id)
