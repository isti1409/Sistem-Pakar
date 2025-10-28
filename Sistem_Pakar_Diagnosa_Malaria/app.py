# app.py
from flask import Flask, render_template, request
from typing import Dict, List, Tuple, Any
import json
import os

app = Flask(__name__)

# Konstanta untuk label keyakinan
CONF_LABELS = [
    ("1.0", "Sangat Yakin"),
    ("0.8", "Yakin"), 
    ("0.5", "Ragu-ragu"),
    ("0.2", "Tidak Yakin"),
    ("0.0", "Sangat Tidak Yakin")
]

class ExpertSystem:
    def __init__(self, kb_file: str = "rules_combined.json"):
        self.kb = self.load_knowledge_base(kb_file)
        
    def load_knowledge_base(self, filename: str) -> dict:
        """Load knowledge base dari file JSON."""
        # Daftar kemungkinan lokasi file
        possible_paths = [
            filename,  # current directory
            os.path.join(os.path.dirname(__file__), filename),  # same dir as app.py
            os.path.join(os.path.dirname(__file__), '..', filename),  # parent dir
            os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)  # absolute path
        ]
        
        for path in possible_paths:
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    print(f"Loading knowledge base from: {path}")
                    return json.load(f)
            except FileNotFoundError:
                continue
            
        # Jika file tidak ditemukan di semua lokasi
        raise Exception(
            f"Knowledge base file {filename} not found. Searched in:\n" + 
            "\n".join(f"- {p}" for p in possible_paths)
        )

    def get_mb_md(self, symptom_code: str) -> Tuple[float, float]:
        """Ambil nilai MB dan MD untuk suatu gejala."""
        for symptom in self.kb.get('symptoms', []):
            if symptom['code'] == symptom_code:
                return float(symptom.get('mb', 1.0)), float(symptom.get('md', 0.0))
        return 1.0, 0.0

    def calculate_cf(self, user_cf: float, symptom_code: str) -> float:
        """Hitung CF akhir dengan mempertimbangkan MB-MD."""
        mb, md = self.get_mb_md(symptom_code)
        base_cf = mb - md
        return user_cf * base_cf

    def combine_cf(self, cf1: float, cf2: float) -> float:
        """Kombinasikan dua CF menggunakan aturan kombinasi MYCIN."""
        if cf1 >= 0 and cf2 >= 0:
            return cf1 + cf2 * (1 - cf1)
        elif cf1 < 0 and cf2 < 0:
            return cf1 + cf2 * (1 + cf1)
        else:
            return (cf1 + cf2) / (1 - min(abs(cf1), abs(cf2)))

    def forward_chaining(self, facts: Dict[str, float]) -> Tuple[Dict[str, float], List[dict]]:
        """
        Implementasi forward chaining dengan CF.
        
        Args:
            facts: Dictionary gejala -> nilai CF user
            
        Returns:
            Tuple dari (cf_facts, trace)
            - cf_facts: Dictionary hasil diagnosa -> nilai CF
            - trace: List trace eksekusi rule
        """
        cf_facts = {}
        trace = []
        
        # Hitung CF awal untuk setiap gejala
        for code, user_cf in facts.items():
            cf_facts[code] = self.calculate_cf(user_cf, code)

        # Iterasi sampai tidak ada rule yang bisa diaplikasikan
        rules_fired = True
        while rules_fired:
            rules_fired = False
            
            for rule in self.kb.get('rules', []):
                rule_id = rule.get('id', '')
                antecedents = rule.get('if', [])
                conclusion = rule.get('then', '')
                rule_cf = float(rule.get('cf', 1.0))
                
                # Skip jika conclusion sudah final
                if conclusion in cf_facts and rule_id in [t.get('rule_id') for t in trace]:
                    continue
                
                # Cek semua antecedent ada
                if not all(ant in cf_facts for ant in antecedents):
                    continue
                
                # Hitung CF rule
                ant_cfs = [cf_facts[ant] for ant in antecedents]
                min_cf = min(ant_cfs)
                rule_contribution = rule_cf * min_cf
                
                # Update conclusion CF
                old_cf = cf_facts.get(conclusion, 0.0)
                if conclusion in cf_facts:
                    new_cf = self.combine_cf(old_cf, rule_contribution)
                else:
                    new_cf = rule_contribution
                
                if abs(new_cf - old_cf) > 1e-6:
                    cf_facts[conclusion] = new_cf
                    rules_fired = True
                    
                    trace.append({
                        'rule_id': rule_id,
                        'if': antecedents,
                        'then': conclusion,
                        'antecedent_cfs': ant_cfs,
                        'rule_cf': rule_cf,
                        'old_cf': old_cf,
                        'new_cf': new_cf,
                        'note': rule.get('note', '')
                    })

        return cf_facts, trace

# Inisialisasi sistem pakar
expert_system = ExpertSystem()

@app.route('/', methods=['GET'])
def index():
    symptoms = expert_system.kb.get('symptoms', [])
    return render_template('index.html', 
                         symptoms=symptoms,
                         conf_labels=CONF_LABELS)

@app.route('/infer', methods=['POST'])
def infer():
    # Ambil input gejala
    selected = request.form.getlist('symptom')
    
    # Konversi ke dictionary gejala -> CF
    input_facts = {}
    input_labels = {}
    for code in selected:
        val = request.form.get(f'confidence_{code}', '1.0')
        try:
            cf_user = float(val)
        except ValueError:
            cf_user = 1.0
        input_facts[code] = cf_user
        label = next((lab for v, lab in CONF_LABELS if float(v) == cf_user), str(cf_user))
        input_labels[code] = label

    # Inference
    cf_facts, trace = expert_system.forward_chaining(input_facts)
    
    # Filter conclusions (non-gejala)
    conclusions = {k: v for k, v in cf_facts.items() if not k.startswith('G')}
    
    return render_template('result.html',
                         input_selected=selected,
                         input_facts=input_facts, 
                         input_labels=input_labels,
                         conclusions=conclusions,
                         trace=trace)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
print("DEBUG: Conclusions:", conclusions)
print("DEBUG: Trace:", trace)
