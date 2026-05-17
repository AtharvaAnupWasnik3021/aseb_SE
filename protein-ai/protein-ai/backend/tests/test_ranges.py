import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from biology_agent import rule_based_assessment
from chemistry_agent import ChemistryAgent
from decision_agent import DecisionAgent
from safety_agent import SafetyAgent
from structure_agent import StructureAgent


class StructureRangeTests(unittest.TestCase):
    def setUp(self):
        self.agent = StructureAgent.__new__(StructureAgent)

    def test_druggability_high_at_optimal_ranges(self):
        score, verdict = self.agent._calculate_druggability(500, 15, 15, 0.4)
        self.assertAlmostEqual(score, 0.8)
        self.assertEqual(verdict, "HIGH DRUGGABILITY (Good target)")

    def test_druggability_low_below_lower_ranges(self):
        score, verdict = self.agent._calculate_druggability(299, 9, 9, 0.39)
        self.assertAlmostEqual(score, 0.14)
        self.assertEqual(verdict, "NOT DRUGGABLE (Poor target)")

    def test_druggability_large_boundary(self):
        score, verdict = self.agent._calculate_druggability(2000, 60, 50, 0.71)
        self.assertAlmostEqual(score, 0.42)
        self.assertEqual(verdict, "LOW DRUGGABILITY (Challenging)")


class BiologyRangeTests(unittest.TestCase):
    def test_biology_verdict_boundaries(self):
        cases = [
            (0.299, "AVOID - insufficient disease link"),
            (0.3, "WEAK TARGET"),
            (0.499, "WEAK TARGET"),
            (0.5, "MODERATE TARGET"),
            (0.699, "MODERATE TARGET"),
            (0.7, "STRONG TARGET"),
        ]
        for score, expected in cases:
            with self.subTest(score=score):
                result = rule_based_assessment(score, {}, "breast cancer")
                self.assertEqual(result["verdict"], expected)

    def test_known_drug_boosts_to_moderate(self):
        result = rule_based_assessment(
            0.2,
            {"Existing drugs target this protein": 0.21},
            "breast cancer",
        )
        self.assertEqual(result["overall_score"], 0.65)
        self.assertEqual(result["verdict"], "MODERATE TARGET")

    def test_clinical_plus_somatic_boosts_to_strong(self):
        result = rule_based_assessment(
            0.6,
            {
                "Clinical trial evidence": 0.51,
                "Somatic mutations in disease tissue": 0.41,
            },
            "breast cancer",
        )
        self.assertEqual(result["overall_score"], 0.75)
        self.assertEqual(result["verdict"], "STRONG TARGET")


class ChemistryRangeTests(unittest.TestCase):
    def test_binding_score_boundaries(self):
        cases = [
            (10, 1.0),
            (10.1, 0.85),
            (100, 0.85),
            (100.1, 0.65),
            (1000, 0.65),
            (1000.1, 0.35),
            (10000, 0.35),
            (10000.1, 0.1),
        ]
        for value, expected in cases:
            with self.subTest(value=value):
                self.assertEqual(ChemistryAgent._binding_score(value), expected)

    def test_property_score_boundaries(self):
        ideal = {"full_mwt": 500, "alogp": 5, "hba": 10, "hbd": 5, "psa": 140}
        outside = {"full_mwt": 501, "alogp": 5.1, "hba": 11, "hbd": 6, "psa": 141}
        self.assertEqual(ChemistryAgent._property_score(ideal), 1.0)
        self.assertEqual(ChemistryAgent._property_score(outside), 0.0)

    def test_chemistry_verdict_boundaries(self):
        agent = ChemistryAgent.__new__(ChemistryAgent)
        ideal = {"full_mwt": 500, "alogp": 5, "hba": 10, "hbd": 5, "psa": 140}
        outside = {"full_mwt": 501, "alogp": 5.1, "hba": 11, "hbd": 6, "psa": 141}
        cases = [
            ("10", ideal, "HIGH FEASIBILITY"),
            ("10000", ideal, "MODERATE FEASIBILITY"),
            ("10001", ideal, "LOW FEASIBILITY"),
            ("10001", outside, "POOR FEASIBILITY"),
        ]
        for value, properties, expected in cases:
            with self.subTest(value=value, expected=expected):
                agent._fetch_molecule = lambda _molecule_id, props=properties: {
                    "molecule_properties": props
                }
                result = agent._score_activities(
                    [{"molecule_chembl_id": "CHEMBL1", "standard_value": value}]
                )
                self.assertEqual(result["chemistry_verdict"], expected)


class SafetyRangeTests(unittest.TestCase):
    def setUp(self):
        self.agent = SafetyAgent("EGFR")

    def test_safety_boundaries(self):
        cases = [
            (0.75, "MODERATE RISK"),
            (0.751, "SAFE"),
            (0.55, "HIGH RISK"),
            (0.551, "MODERATE RISK"),
            (0.35, "UNSAFE"),
            (0.351, "HIGH RISK"),
        ]
        for score, expected in cases:
            with self.subTest(score=score):
                actual, verdict = self.agent.compute(score, None, None)
                self.assertAlmostEqual(actual, score)
                self.assertEqual(verdict, expected)

    def test_missing_components_fall_back_to_unknown(self):
        self.assertEqual(self.agent.compute(None, None, None), (0.5, "UNKNOWN"))


class DecisionRangeTests(unittest.TestCase):
    def setUp(self):
        self.agent = DecisionAgent("2ITX", "EGFR", "breast cancer")
        self.agent.biology_results = {"verdict": "MODERATE TARGET"}
        self.agent.safety_results = {"safety_verdict": "MODERATE RISK"}
        self.agent.chemistry_results = {"chemistry_verdict": "MODERATE FEASIBILITY"}

    def test_final_verdict_boundaries(self):
        cases = [
            (0.349, "REJECT - low integrated confidence"),
            (0.35, "HOLD - borderline target, needs more validation"),
            (0.549, "HOLD - borderline target, needs more validation"),
            (0.55, "CONSIDER - moderate drug target candidate"),
            (0.749, "CONSIDER - moderate drug target candidate"),
            (0.75, "USE - strong drug target candidate"),
        ]
        for score, expected in cases:
            with self.subTest(score=score):
                self.assertEqual(self.agent.final_verdict(score), expected)

    def test_guardrail_rejections_override_score(self):
        self.agent.biology_results = {"verdict": "AVOID - insufficient disease link"}
        self.assertEqual(
            self.agent.final_verdict(0.99),
            "REJECT - weak biological relevance",
        )
        self.agent.biology_results = {"verdict": "STRONG TARGET"}
        self.agent.safety_results = {"safety_verdict": "UNSAFE"}
        self.assertEqual(self.agent.final_verdict(0.99), "REJECT - safety risk")
        self.agent.safety_results = {"safety_verdict": "SAFE"}
        self.agent.chemistry_results = {"chemistry_verdict": "POOR FEASIBILITY"}
        self.assertEqual(
            self.agent.final_verdict(0.99),
            "REJECT - poor chemistry feasibility",
        )


if __name__ == "__main__":
    unittest.main()
