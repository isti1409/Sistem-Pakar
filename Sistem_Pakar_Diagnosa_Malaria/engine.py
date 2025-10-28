# engine.py
import os
import json
from typing import Dict, List, Tuple, Any

RULES_PATHS = ["rules_combined.json", "rules.json"]

# === Load Knowledge Base ===
def load_kb(path: str | None = None) -> Dict[str, Any]:
    """
    Load knowledge base JSON.
    - Mencoba mencari file rules JSON di beberapa lokasi umum.
    """
    base_dir = os.path.dirname(__file__)
    candidates = []

    if path:
        # jika path absolute atau relatif diberikan, coba langsung
        candidates.append(path)
        candidates.append(os.path.join(base_dir, path))

    for name in RULES_PATHS:
        candidates.append(os.path.join(base_dir, name))
        candidates.append(name)

    for p in candidates:
        if not p:
            continue
        try:
            with open(p, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except FileNotFoundError:
            continue
        except json.JSONDecodeError:
            # jika file ada tapi rusak, biarkan error lebih jelas
            raise

    raise FileNotFoundError(
        "File rules JSON tidak ditemukan.\n"
        "Dicari di:\n  " + "\n  ".join(candidates)
    )

# === CF Utility ===
def cf_symptom_from_mbmd(symptom: dict) -> float:
    """Hitung nilai CF dasar dari MB−MD."""
    return float(symptom.get("mb", 0.0)) - float(symptom.get("md", 0.0))

def combine_cf(cf1: float, cf2: float) -> float:
    """Kombinasi CF model MYCIN (tanpa pembulatan berlebih)."""
    cf1 = float(cf1)
    cf2 = float(cf2)
    if cf1 >= 0 and cf2 >= 0:
        return cf1 + cf2 * (1 - cf1)
    if cf1 <= 0 and cf2 <= 0:
        return cf1 + cf2 * (1 + cf1)
    denom = 1 - min(abs(cf1), abs(cf2))
    if denom == 0:
        return 0.0
    return (cf1 + cf2) / denom

# === Forward Chaining dengan CF ===
def forward_chaining(kb: dict,
                     input_facts: Dict[str, float],
                     debug: bool = False) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
    """
    Melakukan inferensi dengan metode forward chaining dan Certainty Factor.
    - input_facts: mapping kode -> confidence pengguna (0..1)
      untuk gejala, nilai dasar = user_confidence * (MB - MD)
    - rules: setiap rule memiliki 'if' (list antecedents), 'then' (conclusion), 'cf' (rule strength)
    """
    # Mapping gejala -> data symptom
    symptom_map = {s["code"]: s for s in kb.get("symptoms", [])}
    base_cf = {code: cf_symptom_from_mbmd(symptom_map[code]) for code in symptom_map}

    if debug:
        print("DEBUG base_cf (MB-MD per symptom):")
        for k, v in sorted(base_cf.items()):
            print(f"  {k}: {v}")

    # Fakta awal berdasarkan input pengguna
    cf_facts: Dict[str, float] = {}
    for code, user_conf in input_facts.items():
        try:
            user_conf = float(user_conf)
        except Exception:
            user_conf = 1.0
        if code in base_cf:
            # untuk gejala: skala MB-MD oleh keyakinan user
            cf_facts[code] = user_conf * base_cf[code]
        else:
            # jika user memasukkan fakta non-gejala, simpan langsung
            cf_facts[code] = user_conf

    if debug:
        print("DEBUG initial cf_facts (user_conf * base_cf):")
        for k, v in sorted(cf_facts.items()):
            print(f"  {k}: {v}")

    rules = kb.get("rules", [])
    fired_trace: List[Dict[str, Any]] = []

    # catat terakhir cf_antecedent yang dipakai tiap rule untuk mencegah re-fire identik
    applied_rules: Dict[str, float] = {}

    changed = True
    iteration = 0
    while changed:
        iteration += 1
        changed = False
        if debug:
            print(f"\nDEBUG iteration {iteration}, existing facts: {len(cf_facts)}")
        for rule in rules:
            rid = rule.get("id")
            antecedents = rule.get("if", [])
            conclusion = rule.get("then")
            rule_strength = float(rule.get("cf", 1.0))

            # ambil CF masing-masing antecedent; jika salah satu tidak ada, rule tidak bisa dipakai
            antecedent_cfs = []
            missing = False
            for a in antecedents:
                if a in cf_facts:
                    antecedent_cfs.append(cf_facts[a])
                else:
                    missing = True
                    break
            if missing or not antecedent_cfs:
                continue

            # untuk AND sederhana gunakan min(antecedent_cfs) (konservatif)
            cf_antecedent = min(antecedent_cfs)

            # jika rule sudah pernah dipakai dengan cf_antecedent yang sama (tidak berubah),
            # skip untuk mencegah penggabungan berulang yang mendorong CF ke 1
            prev_cf = applied_rules.get(rid)
            if prev_cf is not None and abs(prev_cf - cf_antecedent) < 1e-9:
                if debug:
                    print(f"Skip {rid}: antecedent cf unchanged ({cf_antecedent})")
                continue

            # nilai kontribusi rule ke conclusion
            rule_cf_value = rule_strength * cf_antecedent

            old_concl_cf = cf_facts.get(conclusion, 0.0)
            # kombinasi dengan nilai conclusion yang sudah ada (jika ada)
            if conclusion in cf_facts:
                new_concl_cf = combine_cf(old_concl_cf, rule_cf_value)
            else:
                new_concl_cf = rule_cf_value

            # update jika ada perubahan signifikan
            if abs(new_concl_cf - old_concl_cf) > 1e-6:
                cf_facts[conclusion] = new_concl_cf
                changed = True
                applied_rules[rid] = cf_antecedent
                trace_item = {
                    "rule_id": rid,
                    "if": antecedents,
                    "antecedent_cfs": antecedent_cfs,
                    "cf_antecedent": cf_antecedent,
                    "rule_cf": rule_strength,
                    "rule_contribution": rule_cf_value,
                    "conclusion": conclusion,
                    "old_cf": old_concl_cf,
                    "new_cf": new_concl_cf,
                    "note": rule.get("note", "")
                }
                fired_trace.append(trace_item)
                if debug:
                    print(f"Fired {rid}: antecedent_cfs={antecedent_cfs}, rule_cf={rule_strength}, contribution={rule_cf_value}")
                    print(f"  -> {conclusion}: {old_concl_cf} -> {new_concl_cf}")

    # batasi nilai CF agar tidak >1.0 atau <−1.0 dan bulatkan
    for k, v in list(cf_facts.items()):
        cf_facts[k] = max(min(round(v, 4), 1.0), -1.0)
    # === Tambahkan kode ini ===
    translation_map = {
        "Anemia_due_to_malaria": "Anemia akibat malaria",
        "Respiratory_complication": "Komplikasi pernapasan akibat malaria",
        "Malaria Tertiana": "Malaria Tertiana",
        "Malaria Tropika": "Malaria Tropika"
    }

    cf_facts = {translation_map.get(k, k): v for k, v in cf_facts.items()}

    return cf_facts, fired_trace


# === CLI untuk Debug Manual ===
if __name__ == "__main__":
    kb = load_kb()
    # tambahkan G19 untuk memicu R_MTE_support / R_MTE_full
    sample_input = {"G02": 1.0, "G03": 1.0, "G08": 1.0, "G18": 1.0, "G19": 1.0}
    cf_facts, trace = forward_chaining(kb, sample_input, debug=True)

    print("\n=== Hasil Inferensi (Final) ===")
    for k, v in sorted(cf_facts.items()):
        print(f"  {k}: {v}")

    print("\n=== Trace (rule fired) ===")
    for t in trace:
        print(t)
