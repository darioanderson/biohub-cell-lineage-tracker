from pathlib import Path
from streamlit.testing.v1 import AppTest


def test_home_explains_goals_without_old_external_buttons():
    root=Path(__file__).resolve().parents[1]
    app=AppTest.from_file(str(root/'app.py')).run(timeout=30)
    assert not app.exception
    labels=[button.label for button in app.button]
    assert 'Open live app' not in labels and 'View source on GitHub' not in labels
    text=' '.join(x.value for x in app.markdown)
    assert 'Research goal: identify developmental differences' in text
    compare=next(button for button in app.button if button.label=='Compare recordings')
    compare.click().run(timeout=30)
    assert not app.exception
    assert app.session_state['main_nav']=='Compare Lineages'
