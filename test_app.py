"""Headless test of app.py with the user's own stack (master_deploy env).

Simulates: type the CSV path -> click the form's "Run" submit button, then
checks that the table preview rendered, no exception was raised, the metrics
rendered and the progress bar reached 100%.
"""
from streamlit.testing.v1 import AppTest

PATH = "../databases/preprocessed_dfs/rr_sensorspreprocessed_df_full.csv"

at = AppTest.from_file("app.py", default_timeout=1200)
at.run()
print("STEP 1 (initial render) exceptions:", [e.value for e in at.exception])
assert not at.exception, "initial render failed"

# Type the CSV path -> data loads and the single run-form is rendered.
at.text_input[0].set_value(PATH)
at.run()
print("STEP 2 (data loaded) exceptions:", [e.value for e in at.exception])
assert not at.exception, "data render failed (st.dataframe width?)"
assert len(at.dataframe) >= 1, "table preview was not rendered"
assert len(at.form_submit_button) == 1, "expected exactly one form submit button (Run)"
print("   table preview OK; form submit buttons:",
      [b.label for b in at.form_submit_button])

# Click the form's "Run" submit button (commits all widgets + runs the pipeline).
at.form_submit_button[0].set_value(True)
at.run()
print("STEP 3 (run) exceptions:", [e.value for e in at.exception])
assert not at.exception, "run failed"

metrics = [(m.label, m.value) for m in at.metric]
print("METRICS:", metrics)
assert metrics, "metrics were not rendered -> pipeline did not run"

progress = [(round(p.value, 2), p.text) for p in at.progress]
print("PROGRESS BAR:", progress)
assert progress and progress[-1][0] >= 1.0, "progress bar did not reach 100%"
print("APP TEST PASSED")
