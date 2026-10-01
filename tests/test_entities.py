from src.entities.extractor import extract_entities

def test_entities():
    e=extract_entities("Order #A78291 was charged ₹4,999 on 03/05/2026.")
    assert "A78291" in e["order_ids"] and e["amounts"] and e["dates"]
