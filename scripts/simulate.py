"""Entry point — delegates to oculomotor.llm_pipeline.simulate."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Re-export symbols so scripts/server.py can still import from here
from oculomotor.llm_pipeline.simulate import main, call_llm, _call_llm, _call_llm_comparison  # noqa: F401

if __name__ == '__main__':
    main()
