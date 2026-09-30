"""Registry pages: register once, find again, see the timeline."""
from tests.conftest_records import register, rclient  # noqa: F401


def test_every_page_links_the_patient_registry(rclient):
    for path in ["/", "/ecg", "/eeg", "/brain-tumor"]:
        assert 'href="/patients"' in rclient.get(path).get_data(as_text=True), path


def test_register_issues_an_id_and_opens_the_patient(rclient):
    pid = register(rclient)
    assert pid == "PT-000001"
    html = rclient.get(f"/patients/{pid}").get_data(as_text=True)
    assert "Asha Rao" in html and "PT-000001" in html and "Penicillin allergy" in html
    assert "No studies yet" in html


def test_register_without_a_name_shows_an_error(rclient):
    resp = rclient.post("/patients/new", data={"name": ""})
    assert resp.status_code == 200
    assert "Patient name is required" in resp.get_data(as_text=True)


def test_search_finds_by_name_id_and_phone(rclient):
    pid = register(rclient)
    register(rclient, name="Ben Ode", phone="555")
    for q in ["asha", pid, "12345"]:
        html = rclient.get(f"/patients?q={q}").get_data(as_text=True)
        assert pid in html and "Ben Ode" not in html, q


def test_unknown_patient_is_404(rclient):
    resp = rclient.get("/patients/PT-999999")
    assert resp.status_code == 404
    assert 'href="/patients"' in resp.get_data(as_text=True)


def test_patient_page_offers_every_new_study(rclient):
    pid = register(rclient)
    html = rclient.get(f"/patients/{pid}").get_data(as_text=True)
    for mod in ["/ecg", "/eeg", "/brain-tumor"]:
        assert f'href="{mod}?patient={pid}"' in html, mod
