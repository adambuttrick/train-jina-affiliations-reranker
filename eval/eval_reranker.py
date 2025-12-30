# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "sentence-transformers>=4.0.0",
#     "einops",
#     "torch",
# ]
# ///
"""
Evaluation for comparing base Jina reranker vs fine-tuned model.

Tests 368 cases across 10 tiers of increasing difficulty:
1. Baseline - Exact/near-exact matches
2. OCR/Noise - Extraction artifacts, typos, case issues
3. Abbreviations - Formal and informal abbreviations
4. Hierarchical - Department/lab/school structures
5. Medical/Hospital - Healthcare affiliations
6. Research Labs - National labs, corporate research
7. Cross-lingual, transliteration, non-Latin scripts
8. Disambiguation - Confusable institutions
9. Negative Controls - Should NOT match
10. Ultra-Hard - Combined challenges
"""

import json
import statistics
from dataclasses import dataclass
from typing import Optional

from sentence_transformers import CrossEncoder


@dataclass
class RankingTestCase:
    """A single ranking test case."""
    name: str
    tier: str  # tier_1_baseline through tier_10_ultrahard
    anchor: str
    positive: str
    negatives: list[str]
    notes: Optional[str] = None


# TIER 1: BASELINE (30 cases) - Exact/near-exact matches

TIER_1_BASELINE = [
    RankingTestCase(
        name="Harvard exact",
        tier="tier_1_baseline",
        anchor="Harvard University",
        positive="Harvard University",
        negatives=["Yale University", "Princeton University", "Columbia University"],
    ),
    RankingTestCase(
        name="Oxford word reorder",
        tier="tier_1_baseline",
        anchor="University of Oxford",
        positive="Oxford University",
        negatives=["Cambridge University", "Imperial College", "UCL"],
    ),
    RankingTestCase(
        name="Stanford exact",
        tier="tier_1_baseline",
        anchor="Stanford University",
        positive="Stanford University",
        negatives=["UC Berkeley", "Caltech", "USC"],
    ),
    RankingTestCase(
        name="MIT full name",
        tier="tier_1_baseline",
        anchor="Massachusetts Institute of Technology",
        positive="Massachusetts Institute of Technology",
        negatives=["Georgia Tech", "Caltech", "Carnegie Mellon"],
    ),
    RankingTestCase(
        name="Cambridge exact",
        tier="tier_1_baseline",
        anchor="University of Cambridge",
        positive="Cambridge University",
        negatives=["Oxford University", "Imperial College London", "King's College London"],
    ),
    RankingTestCase(
        name="Tokyo with article",
        tier="tier_1_baseline",
        anchor="University of Tokyo",
        positive="The University of Tokyo",
        negatives=["Kyoto University", "Osaka University", "Tohoku University"],
    ),
    RankingTestCase(
        name="ETH umlaut",
        tier="tier_1_baseline",
        anchor="ETH Zurich",
        positive="ETH Zürich",
        negatives=["EPFL", "University of Zurich", "University of Basel"],
    ),
    RankingTestCase(
        name="Caltech full",
        tier="tier_1_baseline",
        anchor="California Institute of Technology",
        positive="California Institute of Technology",
        negatives=["MIT", "Stanford", "Georgia Tech"],
    ),
    RankingTestCase(
        name="Princeton exact",
        tier="tier_1_baseline",
        anchor="Princeton University",
        positive="Princeton University",
        negatives=["Yale University", "Harvard University", "Columbia University"],
    ),
    RankingTestCase(
        name="Berkeley full",
        tier="tier_1_baseline",
        anchor="University of California, Berkeley",
        positive="University of California, Berkeley",
        negatives=["UCLA", "UCSD", "Stanford"],
    ),
    RankingTestCase(
        name="Toronto exact",
        tier="tier_1_baseline",
        anchor="University of Toronto",
        positive="University of Toronto",
        negatives=["McGill University", "University of British Columbia", "McMaster University"],
    ),
    RankingTestCase(
        name="Munich German/English",
        tier="tier_1_baseline",
        anchor="Technical University of Munich",
        positive="Technische Universität München",
        negatives=["LMU Munich", "University of Stuttgart", "KIT"],
    ),
    RankingTestCase(
        name="Yale exact",
        tier="tier_1_baseline",
        anchor="Yale University",
        positive="Yale University",
        negatives=["Harvard University", "Princeton University", "Brown University"],
    ),
    RankingTestCase(
        name="Columbia exact",
        tier="tier_1_baseline",
        anchor="Columbia University",
        positive="Columbia University",
        negatives=["NYU", "Cornell University", "Brown University"],
    ),
    RankingTestCase(
        name="Chicago exact",
        tier="tier_1_baseline",
        anchor="University of Chicago",
        positive="University of Chicago",
        negatives=["Northwestern University", "University of Illinois", "DePaul University"],
    ),
    RankingTestCase(
        name="Penn exact",
        tier="tier_1_baseline",
        anchor="University of Pennsylvania",
        positive="University of Pennsylvania",
        negatives=["Penn State", "Temple University", "Drexel University"],
    ),
    RankingTestCase(
        name="Duke exact",
        tier="tier_1_baseline",
        anchor="Duke University",
        positive="Duke University",
        negatives=["UNC Chapel Hill", "Wake Forest University", "NC State"],
    ),
    RankingTestCase(
        name="Northwestern exact",
        tier="tier_1_baseline",
        anchor="Northwestern University",
        positive="Northwestern University",
        negatives=["University of Chicago", "University of Illinois", "Notre Dame"],
    ),
    RankingTestCase(
        name="Cornell exact",
        tier="tier_1_baseline",
        anchor="Cornell University",
        positive="Cornell University",
        negatives=["Columbia University", "NYU", "Syracuse University"],
    ),
    RankingTestCase(
        name="Brown exact",
        tier="tier_1_baseline",
        anchor="Brown University",
        positive="Brown University",
        negatives=["Harvard University", "Yale University", "Dartmouth College"],
    ),
    RankingTestCase(
        name="Dartmouth exact",
        tier="tier_1_baseline",
        anchor="Dartmouth College",
        positive="Dartmouth College",
        negatives=["Brown University", "Amherst College", "Williams College"],
    ),
    RankingTestCase(
        name="Hopkins exact",
        tier="tier_1_baseline",
        anchor="Johns Hopkins University",
        positive="Johns Hopkins University",
        negatives=["Georgetown University", "George Washington University", "University of Maryland"],
    ),
    RankingTestCase(
        name="McGill exact",
        tier="tier_1_baseline",
        anchor="McGill University",
        positive="McGill University",
        negatives=["University of Toronto", "University of Montreal", "Concordia University"],
    ),
    RankingTestCase(
        name="Kyoto exact",
        tier="tier_1_baseline",
        anchor="Kyoto University",
        positive="Kyoto University",
        negatives=["University of Tokyo", "Osaka University", "Tohoku University"],
    ),
    RankingTestCase(
        name="Melbourne exact",
        tier="tier_1_baseline",
        anchor="University of Melbourne",
        positive="The University of Melbourne",
        negatives=["Monash University", "University of Sydney", "UNSW"],
    ),
    RankingTestCase(
        name="Edinburgh exact",
        tier="tier_1_baseline",
        anchor="University of Edinburgh",
        positive="The University of Edinburgh",
        negatives=["University of Glasgow", "University of St Andrews", "University of Aberdeen"],
    ),
    RankingTestCase(
        name="KU Leuven exact",
        tier="tier_1_baseline",
        anchor="KU Leuven",
        positive="Katholieke Universiteit Leuven",
        negatives=["Ghent University", "Université catholique de Louvain", "VU Brussels"],
    ),
    RankingTestCase(
        name="Amsterdam exact",
        tier="tier_1_baseline",
        anchor="University of Amsterdam",
        positive="Universiteit van Amsterdam",
        negatives=["VU Amsterdam", "Leiden University", "Utrecht University"],
    ),
    RankingTestCase(
        name="Copenhagen exact",
        tier="tier_1_baseline",
        anchor="University of Copenhagen",
        positive="Københavns Universitet",
        negatives=["Aarhus University", "DTU", "Aalborg University"],
    ),
    RankingTestCase(
        name="Heidelberg exact",
        tier="tier_1_baseline",
        anchor="Heidelberg University",
        positive="Ruprecht-Karls-Universität Heidelberg",
        negatives=["LMU Munich", "Humboldt University", "Free University of Berlin"],
    ),
]

# TIER 2: OCR/NOISE (30 cases) - Extraction artifacts

TIER_2_OCR_NOISE = [
    RankingTestCase(
        name="Space in Stanford",
        tier="tier_2_ocr_noise",
        anchor="Stan ford University",
        positive="Stanford University",
        negatives=["UC Berkeley", "Caltech", "USC"],
        notes="OCR space insertion",
    ),
    RankingTestCase(
        name="Missing space MIT",
        tier="tier_2_ocr_noise",
        anchor="MassachusettsInstitute of Technology",
        positive="Massachusetts Institute of Technology",
        negatives=["Georgia Tech", "Caltech", "Carnegie Mellon"],
        notes="OCR missing space",
    ),
    RankingTestCase(
        name="All caps Harvard",
        tier="tier_2_ocr_noise",
        anchor="HARVARD UNIVERSITY",
        positive="Harvard University",
        negatives=["Yale University", "Princeton University", "Columbia University"],
        notes="PDF header extraction",
    ),
    RankingTestCase(
        name="All lowercase MIT",
        tier="tier_2_ocr_noise",
        anchor="massachusetts institute of technology",
        positive="Massachusetts Institute of Technology",
        negatives=["Georgia Tech", "Caltech", "Carnegie Mellon"],
        notes="Lowercased text",
    ),
    RankingTestCase(
        name="Double space Berkeley",
        tier="tier_2_ocr_noise",
        anchor="University of  California, Berkeley",
        positive="University of California, Berkeley",
        negatives=["UCLA", "UCSD", "Stanford"],
        notes="Double space",
    ),
    RankingTestCase(
        name="Truncated Univ.",
        tier="tier_2_ocr_noise",
        anchor="Stanford Univ.",
        positive="Stanford University",
        negatives=["UC Berkeley", "Caltech", "USC"],
        notes="Common abbreviation",
    ),
    RankingTestCase(
        name="Truncated Inst.",
        tier="tier_2_ocr_noise",
        anchor="Mass. Inst. of Technology",
        positive="Massachusetts Institute of Technology",
        negatives=["Georgia Tech", "Caltech", "Carnegie Mellon"],
        notes="Multiple truncations",
    ),
    RankingTestCase(
        name="Merged department text",
        tier="tier_2_ocr_noise",
        anchor="Department of PhysicsStanford University",
        positive="Stanford University",
        negatives=["UC Berkeley", "Caltech", "MIT"],
        notes="Missing line break",
    ),
    RankingTestCase(
        name="Leading space",
        tier="tier_2_ocr_noise",
        anchor=" Harvard University",
        positive="Harvard University",
        negatives=["Yale University", "Princeton University", "Columbia University"],
        notes="Whitespace artifact",
    ),
    RankingTestCase(
        name="Trailing space",
        tier="tier_2_ocr_noise",
        anchor="Harvard University ",
        positive="Harvard University",
        negatives=["Yale University", "Princeton University", "Columbia University"],
        notes="Whitespace artifact",
    ),
    RankingTestCase(
        name="Tab character",
        tier="tier_2_ocr_noise",
        anchor="Harvard\tUniversity",
        positive="Harvard University",
        negatives=["Yale University", "Princeton University", "Columbia University"],
        notes="Tab instead of space",
    ),
    RankingTestCase(
        name="Comma instead of period",
        tier="tier_2_ocr_noise",
        anchor="Univ, of California, Berkeley",
        positive="University of California, Berkeley",
        negatives=["UCLA", "UCSD", "Stanford"],
        notes="OCR punctuation error",
    ),
    RankingTestCase(
        name="Missing vowel",
        tier="tier_2_ocr_noise",
        anchor="Stanfrd University",
        positive="Stanford University",
        negatives=["UC Berkeley", "Caltech", "USC"],
        notes="OCR missing character",
    ),
    RankingTestCase(
        name="l for I confusion",
        tier="tier_2_ocr_noise",
        anchor="Massachusetls Institute of Technology",
        positive="Massachusetts Institute of Technology",
        negatives=["Georgia Tech", "Caltech", "Carnegie Mellon"],
        notes="OCR l/I confusion",
    ),
    RankingTestCase(
        name="rn for m confusion",
        tier="tier_2_ocr_noise",
        anchor="Colurnbia University",
        positive="Columbia University",
        negatives=["NYU", "Cornell University", "Brown University"],
        notes="OCR rn/m confusion",
    ),
    RankingTestCase(
        name="0 for O confusion",
        tier="tier_2_ocr_noise",
        anchor="University 0f Oxford",
        positive="University of Oxford",
        negatives=["Cambridge University", "Imperial College", "UCL"],
        notes="OCR 0/O confusion",
    ),
    RankingTestCase(
        name="1 for l confusion",
        tier="tier_2_ocr_noise",
        anchor="Ca1tech",
        positive="California Institute of Technology",
        negatives=["MIT", "Stanford", "Georgia Tech"],
        notes="OCR 1/l confusion",
    ),
    RankingTestCase(
        name="Mixed case chaos",
        tier="tier_2_ocr_noise",
        anchor="uNIVERSITY oF cAMBRIDGE",
        positive="University of Cambridge",
        negatives=["Oxford University", "Imperial College London", "King's College London"],
        notes="Case inversion",
    ),
    RankingTestCase(
        name="Word truncation",
        tier="tier_2_ocr_noise",
        anchor="Univ of Calif Berkeley",
        positive="University of California, Berkeley",
        negatives=["UCLA", "UCSD", "Stanford"],
        notes="Multiple word truncations",
    ),
    RankingTestCase(
        name="Hyphenated extraction",
        tier="tier_2_ocr_noise",
        anchor="Har-vard University",
        positive="Harvard University",
        negatives=["Yale University", "Princeton University", "Columbia University"],
        notes="Line break hyphen retained",
    ),
    RankingTestCase(
        name="Soft hyphen",
        tier="tier_2_ocr_noise",
        anchor="Uni\u00adversity of Toronto",
        positive="University of Toronto",
        negatives=["McGill University", "University of British Columbia", "McMaster University"],
        notes="Unicode soft hyphen (U+00AD)",
    ),
    RankingTestCase(
        name="Footnote marker",
        tier="tier_2_ocr_noise",
        anchor="Stanford University†",
        positive="Stanford University",
        negatives=["UC Berkeley", "Caltech", "USC"],
        notes="Footnote symbol retained",
    ),
    RankingTestCase(
        name="Superscript marker",
        tier="tier_2_ocr_noise",
        anchor="Harvard University¹",
        positive="Harvard University",
        negatives=["Yale University", "Princeton University", "Columbia University"],
        notes="Author affiliation superscript",
    ),
    RankingTestCase(
        name="Email artifact",
        tier="tier_2_ocr_noise",
        anchor="Stanford University, email: john@stanford.edu",
        positive="Stanford University",
        negatives=["UC Berkeley", "Caltech", "USC"],
        notes="Email retained in string",
    ),
    RankingTestCase(
        name="Parenthetical noise",
        tier="tier_2_ocr_noise",
        anchor="Harvard University (Cambridge, MA)",
        positive="Harvard University",
        negatives=["Yale University", "Princeton University", "Columbia University"],
        notes="Location in parens",
    ),
    RankingTestCase(
        name="Country in parens",
        tier="tier_2_ocr_noise",
        anchor="University of Oxford (UK)",
        positive="University of Oxford",
        negatives=["Cambridge University", "Imperial College", "UCL"],
        notes="Country in parens",
    ),
    RankingTestCase(
        name="Date appended",
        tier="tier_2_ocr_noise",
        anchor="MIT (2024)",
        positive="Massachusetts Institute of Technology",
        negatives=["Georgia Tech", "Caltech", "Carnegie Mellon"],
        notes="Year appended",
    ),
    RankingTestCase(
        name="HTML entity ampersand",
        tier="tier_2_ocr_noise",
        anchor="Arts &amp; Sciences, Harvard",
        positive="Harvard University",
        negatives=["Yale University", "Princeton University", "Columbia University"],
        notes="HTML entity not decoded",
    ),
    RankingTestCase(
        name="Smart quotes",
        tier="tier_2_ocr_noise",
        anchor="King's College London",
        positive="King's College London",
        negatives=["UCL", "Imperial College London", "LSE"],
        notes="Curly apostrophe",
    ),
    RankingTestCase(
        name="Asterisk for footnote",
        tier="tier_2_ocr_noise",
        anchor="*Stanford University",
        positive="Stanford University",
        negatives=["UC Berkeley", "Caltech", "USC"],
        notes="Leading asterisk",
    ),
]

# TIER 3: ABBREVIATIONS (40 cases) - Formal and informal

TIER_3_ABBREVIATIONS = [
    RankingTestCase(
        name="MIT standard",
        tier="tier_3_abbreviations",
        anchor="Massachusetts Institute of Technology",
        positive="MIT",
        negatives=["Caltech", "Georgia Tech", "Carnegie Mellon"],
    ),
    RankingTestCase(
        name="UCLA standard",
        tier="tier_3_abbreviations",
        anchor="University of California, Los Angeles",
        positive="UCLA",
        negatives=["USC", "UC Berkeley", "UCSD"],
    ),
    RankingTestCase(
        name="UCL standard",
        tier="tier_3_abbreviations",
        anchor="University College London",
        positive="UCL",
        negatives=["Imperial College", "King's College London", "LSE"],
    ),
    RankingTestCase(
        name="LSE standard",
        tier="tier_3_abbreviations",
        anchor="London School of Economics and Political Science",
        positive="LSE",
        negatives=["UCL", "Imperial College", "King's College London"],
    ),
    RankingTestCase(
        name="CMU standard",
        tier="tier_3_abbreviations",
        anchor="Carnegie Mellon University",
        positive="CMU",
        negatives=["MIT", "Stanford", "Georgia Tech"],
    ),
    RankingTestCase(
        name="NYU standard",
        tier="tier_3_abbreviations",
        anchor="New York University",
        positive="NYU",
        negatives=["Columbia University", "CUNY", "The New School"],
    ),
    RankingTestCase(
        name="USC standard",
        tier="tier_3_abbreviations",
        anchor="University of Southern California",
        positive="USC",
        negatives=["UCLA", "Stanford", "Caltech"],
    ),
    RankingTestCase(
        name="UCSD standard",
        tier="tier_3_abbreviations",
        anchor="University of California, San Diego",
        positive="UCSD",
        negatives=["UCLA", "UC Berkeley", "Stanford"],
    ),
    RankingTestCase(
        name="UCSF standard",
        tier="tier_3_abbreviations",
        anchor="University of California, San Francisco",
        positive="UCSF",
        negatives=["Stanford", "UC Berkeley", "UCLA"],
    ),
    RankingTestCase(
        name="UCI standard",
        tier="tier_3_abbreviations",
        anchor="University of California, Irvine",
        positive="UCI",
        negatives=["UCLA", "UCSD", "USC"],
    ),
    RankingTestCase(
        name="UC Davis standard",
        tier="tier_3_abbreviations",
        anchor="University of California, Davis",
        positive="UC Davis",
        negatives=["UC Berkeley", "UCLA", "UCSD"],
    ),
    RankingTestCase(
        name="UC Santa Barbara",
        tier="tier_3_abbreviations",
        anchor="University of California, Santa Barbara",
        positive="UCSB",
        negatives=["UCLA", "UC Berkeley", "Stanford"],
    ),
    RankingTestCase(
        name="Caltech abbreviation",
        tier="tier_3_abbreviations",
        anchor="California Institute of Technology",
        positive="Caltech",
        negatives=["MIT", "Stanford", "Georgia Tech"],
    ),
    RankingTestCase(
        name="Georgia Tech abbreviation",
        tier="tier_3_abbreviations",
        anchor="Georgia Institute of Technology",
        positive="Georgia Tech",
        negatives=["MIT", "Caltech", "Virginia Tech"],
    ),
    RankingTestCase(
        name="Virginia Tech abbreviation",
        tier="tier_3_abbreviations",
        anchor="Virginia Polytechnic Institute and State University",
        positive="Virginia Tech",
        negatives=["Georgia Tech", "MIT", "NC State"],
    ),
    RankingTestCase(
        name="UPenn abbreviation",
        tier="tier_3_abbreviations",
        anchor="University of Pennsylvania",
        positive="UPenn",
        negatives=["Penn State", "Princeton", "Temple University"],
    ),
    RankingTestCase(
        name="Penn informal",
        tier="tier_3_abbreviations",
        anchor="University of Pennsylvania",
        positive="Penn",
        negatives=["Penn State", "Princeton", "Temple University"],
    ),
    RankingTestCase(
        name="EPFL abbreviation",
        tier="tier_3_abbreviations",
        anchor="École Polytechnique Fédérale de Lausanne",
        positive="EPFL",
        negatives=["ETH Zurich", "University of Geneva", "University of Lausanne"],
    ),
    RankingTestCase(
        name="ETH abbreviation",
        tier="tier_3_abbreviations",
        anchor="Eidgenössische Technische Hochschule Zürich",
        positive="ETH Zurich",
        negatives=["EPFL", "University of Zurich", "University of Basel"],
    ),
    RankingTestCase(
        name="RWTH abbreviation",
        tier="tier_3_abbreviations",
        anchor="Rheinisch-Westfälische Technische Hochschule Aachen",
        positive="RWTH Aachen",
        negatives=["TU Munich", "KIT", "TU Berlin"],
    ),
    RankingTestCase(
        name="TUM abbreviation",
        tier="tier_3_abbreviations",
        anchor="Technical University of Munich",
        positive="TUM",
        negatives=["LMU Munich", "University of Stuttgart", "RWTH Aachen"],
    ),
    RankingTestCase(
        name="KIT abbreviation",
        tier="tier_3_abbreviations",
        anchor="Karlsruhe Institute of Technology",
        positive="KIT",
        negatives=["TU Munich", "RWTH Aachen", "University of Stuttgart"],
    ),
    RankingTestCase(
        name="NUS abbreviation",
        tier="tier_3_abbreviations",
        anchor="National University of Singapore",
        positive="NUS",
        negatives=["NTU Singapore", "SMU Singapore", "SUTD"],
    ),
    RankingTestCase(
        name="NTU Singapore",
        tier="tier_3_abbreviations",
        anchor="Nanyang Technological University",
        positive="NTU",
        negatives=["NUS", "SMU Singapore", "SUTD"],
    ),
    RankingTestCase(
        name="KAIST abbreviation",
        tier="tier_3_abbreviations",
        anchor="Korea Advanced Institute of Science and Technology",
        positive="KAIST",
        negatives=["Seoul National University", "POSTECH", "Korea University"],
    ),
    RankingTestCase(
        name="POSTECH abbreviation",
        tier="tier_3_abbreviations",
        anchor="Pohang University of Science and Technology",
        positive="POSTECH",
        negatives=["KAIST", "Seoul National University", "Korea University"],
    ),
    RankingTestCase(
        name="SNU abbreviation",
        tier="tier_3_abbreviations",
        anchor="Seoul National University",
        positive="SNU",
        negatives=["Korea University", "Yonsei University", "KAIST"],
    ),
    RankingTestCase(
        name="HKUST abbreviation",
        tier="tier_3_abbreviations",
        anchor="Hong Kong University of Science and Technology",
        positive="HKUST",
        negatives=["HKU", "CUHK", "City University of Hong Kong"],
    ),
    RankingTestCase(
        name="HKU abbreviation",
        tier="tier_3_abbreviations",
        anchor="University of Hong Kong",
        positive="HKU",
        negatives=["HKUST", "CUHK", "City University of Hong Kong"],
    ),
    RankingTestCase(
        name="CUHK abbreviation",
        tier="tier_3_abbreviations",
        anchor="Chinese University of Hong Kong",
        positive="CUHK",
        negatives=["HKU", "HKUST", "City University of Hong Kong"],
    ),
    RankingTestCase(
        name="UBC abbreviation",
        tier="tier_3_abbreviations",
        anchor="University of British Columbia",
        positive="UBC",
        negatives=["University of Toronto", "McGill University", "University of Alberta"],
    ),
    RankingTestCase(
        name="ANU abbreviation",
        tier="tier_3_abbreviations",
        anchor="Australian National University",
        positive="ANU",
        negatives=["University of Melbourne", "University of Sydney", "UNSW"],
    ),
    RankingTestCase(
        name="UNSW abbreviation",
        tier="tier_3_abbreviations",
        anchor="University of New South Wales",
        positive="UNSW",
        negatives=["University of Sydney", "University of Melbourne", "ANU"],
    ),
    RankingTestCase(
        name="ENS abbreviation",
        tier="tier_3_abbreviations",
        anchor="École normale supérieure",
        positive="ENS Paris",
        negatives=["École Polytechnique", "Sorbonne University", "Paris-Saclay"],
    ),
    RankingTestCase(
        name="Sciences Po abbreviation",
        tier="tier_3_abbreviations",
        anchor="Institut d'études politiques de Paris",
        positive="Sciences Po",
        negatives=["Sorbonne University", "ENS Paris", "HEC Paris"],
    ),
    RankingTestCase(
        name="Cal informal",
        tier="tier_3_abbreviations",
        anchor="University of California, Berkeley",
        positive="Cal",
        negatives=["UCLA", "Stanford", "USC"],
        notes="Informal Berkeley name",
    ),
    RankingTestCase(
        name="CSAIL lab name",
        tier="tier_3_abbreviations",
        anchor="MIT Computer Science and Artificial Intelligence Laboratory",
        positive="MIT CSAIL",
        negatives=["Stanford AI Lab", "CMU Robotics", "Berkeley AI"],
    ),
    RankingTestCase(
        name="Stanford AI Lab",
        tier="tier_3_abbreviations",
        anchor="Stanford Artificial Intelligence Laboratory",
        positive="SAIL",
        negatives=["MIT CSAIL", "CMU Robotics", "Berkeley AI"],
    ),
    RankingTestCase(
        name="UT Austin",
        tier="tier_3_abbreviations",
        anchor="University of Texas at Austin",
        positive="UT Austin",
        negatives=["Texas A&M", "Rice University", "UT Dallas"],
    ),
    RankingTestCase(
        name="UMass Amherst",
        tier="tier_3_abbreviations",
        anchor="University of Massachusetts Amherst",
        positive="UMass Amherst",
        negatives=["UMass Boston", "Boston University", "Northeastern"],
    ),
]

# TIER 4: HIERARCHICAL (35 cases) - Department/lab structures

TIER_4_HIERARCHICAL = [
    RankingTestCase(
        name="CS dept Stanford",
        tier="tier_4_hierarchical",
        anchor="Department of Computer Science, Stanford University",
        positive="Stanford University",
        negatives=["MIT", "UC Berkeley", "Carnegie Mellon"],
    ),
    RankingTestCase(
        name="Physics MIT",
        tier="tier_4_hierarchical",
        anchor="Department of Physics, MIT",
        positive="Massachusetts Institute of Technology",
        negatives=["Caltech", "Princeton", "Harvard"],
    ),
    RankingTestCase(
        name="Math Oxford",
        tier="tier_4_hierarchical",
        anchor="Mathematical Institute, University of Oxford",
        positive="Oxford University",
        negatives=["Cambridge University", "Imperial College", "Warwick University"],
    ),
    RankingTestCase(
        name="EECS Berkeley",
        tier="tier_4_hierarchical",
        anchor="EECS Department, UC Berkeley",
        positive="University of California, Berkeley",
        negatives=["Stanford", "MIT", "CMU"],
    ),
    RankingTestCase(
        name="Inverted order",
        tier="tier_4_hierarchical",
        anchor="Stanford University, Department of Computer Science",
        positive="Stanford University",
        negatives=["MIT", "UC Berkeley", "Carnegie Mellon"],
    ),
    RankingTestCase(
        name="Deep nesting",
        tier="tier_4_hierarchical",
        anchor="AI Lab, Computer Science Dept., School of Engineering, Stanford University",
        positive="Stanford University",
        negatives=["MIT", "UC Berkeley", "Carnegie Mellon"],
    ),
    RankingTestCase(
        name="Building name",
        tier="tier_4_hierarchical",
        anchor="Gates Computer Science Building, Stanford",
        positive="Stanford University",
        negatives=["MIT", "UC Berkeley", "Carnegie Mellon"],
        notes="Building name included",
    ),
    RankingTestCase(
        name="Research center",
        tier="tier_4_hierarchical",
        anchor="Center for Data Science, New York University",
        positive="NYU",
        negatives=["Columbia University", "Cornell", "Princeton"],
    ),
    RankingTestCase(
        name="Institute within university",
        tier="tier_4_hierarchical",
        anchor="Kavli Institute for Theoretical Physics, UCSB",
        positive="University of California, Santa Barbara",
        negatives=["UCLA", "UC Berkeley", "Stanford"],
    ),
    RankingTestCase(
        name="Lab name only",
        tier="tier_4_hierarchical",
        anchor="The Robotics Institute",
        positive="Carnegie Mellon University",
        negatives=["MIT", "Stanford", "Georgia Tech"],
        notes="Famous CMU lab",
    ),
    RankingTestCase(
        name="School of Engineering",
        tier="tier_4_hierarchical",
        anchor="School of Engineering, Stanford University",
        positive="Stanford University",
        negatives=["MIT", "UC Berkeley", "Caltech"],
    ),
    RankingTestCase(
        name="Graduate school",
        tier="tier_4_hierarchical",
        anchor="Graduate School of Business, Stanford University",
        positive="Stanford University",
        negatives=["Harvard", "MIT", "Wharton"],
    ),
    RankingTestCase(
        name="Kennedy School",
        tier="tier_4_hierarchical",
        anchor="Harvard Kennedy School",
        positive="Harvard University",
        negatives=["Georgetown University", "Princeton", "Columbia"],
    ),
    RankingTestCase(
        name="Wharton School",
        tier="tier_4_hierarchical",
        anchor="The Wharton School",
        positive="University of Pennsylvania",
        negatives=["Harvard Business School", "Stanford GSB", "MIT Sloan"],
    ),
    RankingTestCase(
        name="MIT Sloan",
        tier="tier_4_hierarchical",
        anchor="MIT Sloan School of Management",
        positive="Massachusetts Institute of Technology",
        negatives=["Harvard Business School", "Stanford GSB", "Wharton"],
    ),
    RankingTestCase(
        name="Berkeley Haas",
        tier="tier_4_hierarchical",
        anchor="Haas School of Business, UC Berkeley",
        positive="University of California, Berkeley",
        negatives=["Stanford GSB", "UCLA Anderson", "MIT Sloan"],
    ),
    RankingTestCase(
        name="Columbia Business",
        tier="tier_4_hierarchical",
        anchor="Columbia Business School",
        positive="Columbia University",
        negatives=["NYU Stern", "Wharton", "Harvard Business School"],
    ),
    RankingTestCase(
        name="Chicago Booth",
        tier="tier_4_hierarchical",
        anchor="Booth School of Business",
        positive="University of Chicago",
        negatives=["Northwestern Kellogg", "MIT Sloan", "Wharton"],
    ),
    RankingTestCase(
        name="Kellogg School",
        tier="tier_4_hierarchical",
        anchor="Kellogg School of Management",
        positive="Northwestern University",
        negatives=["Chicago Booth", "MIT Sloan", "Harvard Business School"],
    ),
    RankingTestCase(
        name="Media Lab",
        tier="tier_4_hierarchical",
        anchor="MIT Media Lab",
        positive="Massachusetts Institute of Technology",
        negatives=["Stanford", "CMU", "NYU"],
    ),
    RankingTestCase(
        name="Lincoln Lab",
        tier="tier_4_hierarchical",
        anchor="MIT Lincoln Laboratory",
        positive="Massachusetts Institute of Technology",
        negatives=["Stanford", "Caltech", "Georgia Tech"],
    ),
    RankingTestCase(
        name="Jet Propulsion Lab",
        tier="tier_4_hierarchical",
        anchor="Jet Propulsion Laboratory",
        positive="California Institute of Technology",
        negatives=["MIT", "Stanford", "NASA Ames"],
        notes="NASA JPL managed by Caltech",
    ),
    RankingTestCase(
        name="Woods Hole",
        tier="tier_4_hierarchical",
        anchor="Woods Hole Oceanographic Institution",
        positive="MIT",
        negatives=["Harvard", "Boston University", "URI"],
        notes="Joint MIT/WHOI program",
    ),
    RankingTestCase(
        name="Department abbrev",
        tier="tier_4_hierarchical",
        anchor="Dept. of CS, MIT",
        positive="Massachusetts Institute of Technology",
        negatives=["Stanford", "CMU", "Berkeley"],
    ),
    RankingTestCase(
        name="Lab abbrev",
        tier="tier_4_hierarchical",
        anchor="CS Dept, Stanford",
        positive="Stanford University",
        negatives=["MIT", "CMU", "Berkeley"],
    ),
    RankingTestCase(
        name="Mixed hierarchy",
        tier="tier_4_hierarchical",
        anchor="NLP Group, Computer Science, Stanford",
        positive="Stanford University",
        negatives=["MIT", "CMU", "Berkeley"],
    ),
    RankingTestCase(
        name="Institute for Advanced Study",
        tier="tier_4_hierarchical",
        anchor="Institute for Advanced Study",
        positive="Princeton University",
        negatives=["Harvard", "MIT", "Yale"],
        notes="IAS is near Princeton but independent",
    ),
    RankingTestCase(
        name="Santa Fe Institute",
        tier="tier_4_hierarchical",
        anchor="Santa Fe Institute",
        positive="Santa Fe Institute",
        negatives=["University of New Mexico", "Arizona State", "Los Alamos"],
        notes="Independent research institute",
    ),
    RankingTestCase(
        name="Perimeter Institute",
        tier="tier_4_hierarchical",
        anchor="Perimeter Institute for Theoretical Physics",
        positive="University of Waterloo",
        negatives=["University of Toronto", "McGill", "McMaster"],
        notes="PI affiliated with Waterloo",
    ),
    RankingTestCase(
        name="Max Planck Institute",
        tier="tier_4_hierarchical",
        anchor="Max Planck Institute for Astrophysics",
        positive="Max-Planck-Institut für Astrophysik",
        negatives=["ESO", "CERN", "Fermilab"],
    ),
    RankingTestCase(
        name="Fraunhofer Institute",
        tier="tier_4_hierarchical",
        anchor="Fraunhofer Institute for Computer Graphics",
        positive="Fraunhofer Society",
        negatives=["Max Planck Society", "TU Darmstadt", "TU Munich"],
    ),
    RankingTestCase(
        name="Research group",
        tier="tier_4_hierarchical",
        anchor="Vision Lab, Stanford AI Lab",
        positive="Stanford University",
        negatives=["MIT", "CMU", "Berkeley"],
    ),
    RankingTestCase(
        name="Clinical department",
        tier="tier_4_hierarchical",
        anchor="Department of Radiology, Massachusetts General Hospital",
        positive="Harvard University",
        negatives=["Boston University", "Tufts", "MIT"],
    ),
    RankingTestCase(
        name="Faculty affiliation",
        tier="tier_4_hierarchical",
        anchor="Faculty of Arts and Sciences, Harvard",
        positive="Harvard University",
        negatives=["Yale", "Princeton", "Columbia"],
    ),
    RankingTestCase(
        name="College within university",
        tier="tier_4_hierarchical",
        anchor="College of Engineering, University of Michigan",
        positive="University of Michigan",
        negatives=["Michigan State", "Wayne State", "Purdue"],
    ),
]

# TIER 5: MEDICAL/HOSPITAL (25 cases) - Healthcare affiliations

TIER_5_MEDICAL = [
    RankingTestCase(
        name="Harvard Medical School",
        tier="tier_5_medical",
        anchor="Harvard Medical School",
        positive="Harvard University",
        negatives=["Boston University", "Tufts University", "MIT"],
    ),
    RankingTestCase(
        name="Stanford Medicine",
        tier="tier_5_medical",
        anchor="Stanford School of Medicine",
        positive="Stanford University",
        negatives=["UCSF", "UCLA", "UC Berkeley"],
    ),
    RankingTestCase(
        name="Johns Hopkins Medicine",
        tier="tier_5_medical",
        anchor="Johns Hopkins School of Medicine",
        positive="Johns Hopkins University",
        negatives=["University of Maryland", "Georgetown", "George Washington"],
    ),
    RankingTestCase(
        name="Yale School of Medicine",
        tier="tier_5_medical",
        anchor="Yale School of Medicine",
        positive="Yale University",
        negatives=["Harvard", "Columbia", "Brown"],
    ),
    RankingTestCase(
        name="Penn Medicine",
        tier="tier_5_medical",
        anchor="Perelman School of Medicine",
        positive="University of Pennsylvania",
        negatives=["Temple", "Drexel", "Thomas Jefferson"],
    ),
    RankingTestCase(
        name="UCSF Medical",
        tier="tier_5_medical",
        anchor="UCSF School of Medicine",
        positive="University of California, San Francisco",
        negatives=["Stanford", "UCLA", "UC Berkeley"],
    ),
    RankingTestCase(
        name="Mass General Hospital",
        tier="tier_5_medical",
        anchor="Massachusetts General Hospital",
        positive="Harvard University",
        negatives=["Boston University", "Tufts", "MIT"],
        notes="MGH is Harvard-affiliated",
    ),
    RankingTestCase(
        name="Brigham and Women's",
        tier="tier_5_medical",
        anchor="Brigham and Women's Hospital",
        positive="Harvard University",
        negatives=["Boston University", "Tufts", "MIT"],
    ),
    RankingTestCase(
        name="Dana-Farber",
        tier="tier_5_medical",
        anchor="Dana-Farber Cancer Institute",
        positive="Harvard University",
        negatives=["Boston University", "MIT", "Tufts"],
    ),
    RankingTestCase(
        name="Children's Hospital Boston",
        tier="tier_5_medical",
        anchor="Boston Children's Hospital",
        positive="Harvard University",
        negatives=["Boston University", "Tufts", "MIT"],
    ),
    RankingTestCase(
        name="Beth Israel Deaconess",
        tier="tier_5_medical",
        anchor="Beth Israel Deaconess Medical Center",
        positive="Harvard University",
        negatives=["Boston University", "Tufts", "MIT"],
    ),
    RankingTestCase(
        name="Stanford Hospital",
        tier="tier_5_medical",
        anchor="Stanford Hospital and Clinics",
        positive="Stanford University",
        negatives=["UCSF", "UCLA", "USC"],
    ),
    RankingTestCase(
        name="UCLA Medical Center",
        tier="tier_5_medical",
        anchor="UCLA Medical Center",
        positive="University of California, Los Angeles",
        negatives=["USC", "Stanford", "UCSD"],
    ),
    RankingTestCase(
        name="MSKCC affiliation",
        tier="tier_5_medical",
        anchor="Memorial Sloan Kettering Cancer Center",
        positive="Cornell University",
        negatives=["Columbia", "NYU", "Rockefeller"],
        notes="MSK/Cornell/Rockefeller affiliation",
    ),
    RankingTestCase(
        name="Mayo Clinic",
        tier="tier_5_medical",
        anchor="Mayo Clinic",
        positive="Mayo Clinic",
        negatives=["University of Minnesota", "Johns Hopkins", "Cleveland Clinic"],
        notes="Mayo is independent",
    ),
    RankingTestCase(
        name="Cleveland Clinic",
        tier="tier_5_medical",
        anchor="Cleveland Clinic",
        positive="Cleveland Clinic",
        negatives=["Case Western Reserve", "Ohio State", "Mayo Clinic"],
        notes="Cleveland Clinic is independent",
    ),
    RankingTestCase(
        name="MD Anderson",
        tier="tier_5_medical",
        anchor="MD Anderson Cancer Center",
        positive="University of Texas",
        negatives=["Baylor", "Rice", "Texas A&M"],
    ),
    RankingTestCase(
        name="Cedars-Sinai",
        tier="tier_5_medical",
        anchor="Cedars-Sinai Medical Center",
        positive="UCLA",
        negatives=["USC", "Stanford", "UCSD"],
        notes="UCLA affiliation",
    ),
    RankingTestCase(
        name="Mount Sinai",
        tier="tier_5_medical",
        anchor="Icahn School of Medicine at Mount Sinai",
        positive="Mount Sinai",
        negatives=["NYU", "Columbia", "Cornell"],
    ),
    RankingTestCase(
        name="VA Hospital affiliation",
        tier="tier_5_medical",
        anchor="VA Boston Healthcare System",
        positive="Boston University",
        negatives=["Harvard", "Tufts", "MIT"],
        notes="VA affiliations vary",
    ),
    RankingTestCase(
        name="NIH Clinical Center",
        tier="tier_5_medical",
        anchor="NIH Clinical Center",
        positive="National Institutes of Health",
        negatives=["Johns Hopkins", "Georgetown", "George Washington"],
    ),
    RankingTestCase(
        name="HMS teaching hospital",
        tier="tier_5_medical",
        anchor="MGH Department of Medicine",
        positive="Harvard Medical School",
        negatives=["Boston University", "Tufts", "MIT"],
    ),
    RankingTestCase(
        name="Oxford NHS Trust",
        tier="tier_5_medical",
        anchor="Oxford University Hospitals NHS Foundation Trust",
        positive="University of Oxford",
        negatives=["Cambridge", "Imperial College", "UCL"],
    ),
    RankingTestCase(
        name="Karolinska",
        tier="tier_5_medical",
        anchor="Karolinska Institutet",
        positive="Karolinska Institutet",
        negatives=["Uppsala University", "Lund University", "Stockholm University"],
        notes="Independent medical university",
    ),
    RankingTestCase(
        name="Charité Berlin",
        tier="tier_5_medical",
        anchor="Charité – Universitätsmedizin Berlin",
        positive="Free University of Berlin",
        negatives=["Humboldt University", "TU Berlin", "University of Potsdam"],
        notes="Charité is joint FU/HU",
    ),
]

# TIER 6: RESEARCH LABS (25 cases) - National labs, corporate research

TIER_6_RESEARCH_LABS = [
    RankingTestCase(
        name="SLAC National Lab",
        tier="tier_6_research_labs",
        anchor="SLAC National Accelerator Laboratory",
        positive="Stanford University",
        negatives=["UC Berkeley", "Caltech", "MIT"],
        notes="SLAC managed by Stanford",
    ),
    RankingTestCase(
        name="Lawrence Berkeley Lab",
        tier="tier_6_research_labs",
        anchor="Lawrence Berkeley National Laboratory",
        positive="UC Berkeley",
        negatives=["Stanford", "LLNL", "Caltech"],
        notes="LBNL managed by UC",
    ),
    RankingTestCase(
        name="Los Alamos",
        tier="tier_6_research_labs",
        anchor="Los Alamos National Laboratory",
        positive="Los Alamos National Laboratory",
        negatives=["Sandia", "Lawrence Livermore", "University of New Mexico"],
        notes="LANL managed by Triad",
    ),
    RankingTestCase(
        name="Sandia National Labs",
        tier="tier_6_research_labs",
        anchor="Sandia National Laboratories",
        positive="Sandia National Laboratories",
        negatives=["Los Alamos", "Lawrence Livermore", "Oak Ridge"],
    ),
    RankingTestCase(
        name="Oak Ridge",
        tier="tier_6_research_labs",
        anchor="Oak Ridge National Laboratory",
        positive="Oak Ridge National Laboratory",
        negatives=["University of Tennessee", "Vanderbilt", "Los Alamos"],
    ),
    RankingTestCase(
        name="Argonne National Lab",
        tier="tier_6_research_labs",
        anchor="Argonne National Laboratory",
        positive="University of Chicago",
        negatives=["Northwestern", "Illinois Tech", "Fermilab"],
        notes="Argonne managed by UChicago",
    ),
    RankingTestCase(
        name="Fermilab",
        tier="tier_6_research_labs",
        anchor="Fermi National Accelerator Laboratory",
        positive="Fermilab",
        negatives=["University of Chicago", "Argonne", "CERN"],
    ),
    RankingTestCase(
        name="Brookhaven",
        tier="tier_6_research_labs",
        anchor="Brookhaven National Laboratory",
        positive="Stony Brook University",
        negatives=["Columbia", "Cornell", "NYU"],
        notes="BNL managed by BSA (Stony Brook)",
    ),
    RankingTestCase(
        name="NIST",
        tier="tier_6_research_labs",
        anchor="National Institute of Standards and Technology",
        positive="NIST",
        negatives=["University of Maryland", "Johns Hopkins", "George Washington"],
    ),
    RankingTestCase(
        name="NIH",
        tier="tier_6_research_labs",
        anchor="National Institutes of Health",
        positive="NIH",
        negatives=["Johns Hopkins", "Georgetown", "University of Maryland"],
    ),
    RankingTestCase(
        name="NASA Ames",
        tier="tier_6_research_labs",
        anchor="NASA Ames Research Center",
        positive="NASA",
        negatives=["Stanford", "UC Berkeley", "UCSC"],
    ),
    RankingTestCase(
        name="NASA JPL",
        tier="tier_6_research_labs",
        anchor="NASA Jet Propulsion Laboratory",
        positive="Caltech",
        negatives=["MIT", "Stanford", "UCLA"],
    ),
    RankingTestCase(
        name="CERN",
        tier="tier_6_research_labs",
        anchor="CERN",
        positive="European Organization for Nuclear Research",
        negatives=["ETH Zurich", "EPFL", "University of Geneva"],
    ),
    RankingTestCase(
        name="Bell Labs",
        tier="tier_6_research_labs",
        anchor="Bell Laboratories",
        positive="Nokia Bell Labs",
        negatives=["MIT", "Stanford", "Princeton"],
        notes="Now Nokia",
    ),
    RankingTestCase(
        name="IBM Research",
        tier="tier_6_research_labs",
        anchor="IBM Research - Almaden",
        positive="IBM Research",
        negatives=["Stanford", "UC Berkeley", "MIT"],
    ),
    RankingTestCase(
        name="Microsoft Research",
        tier="tier_6_research_labs",
        anchor="Microsoft Research",
        positive="Microsoft Research",
        negatives=["University of Washington", "Stanford", "MIT"],
    ),
    RankingTestCase(
        name="Google Research",
        tier="tier_6_research_labs",
        anchor="Google Research",
        positive="Google",
        negatives=["Stanford", "MIT", "UC Berkeley"],
    ),
    RankingTestCase(
        name="DeepMind",
        tier="tier_6_research_labs",
        anchor="Google DeepMind",
        positive="Google",
        negatives=["Oxford", "UCL", "Cambridge"],
    ),
    RankingTestCase(
        name="Meta AI",
        tier="tier_6_research_labs",
        anchor="Meta AI Research (FAIR)",
        positive="Meta",
        negatives=["NYU", "MIT", "Stanford"],
    ),
    RankingTestCase(
        name="OpenAI",
        tier="tier_6_research_labs",
        anchor="OpenAI",
        positive="OpenAI",
        negatives=["Google", "Microsoft", "MIT"],
    ),
    RankingTestCase(
        name="Allen Institute",
        tier="tier_6_research_labs",
        anchor="Allen Institute for Artificial Intelligence",
        positive="AI2",
        negatives=["University of Washington", "Microsoft Research", "Stanford"],
    ),
    RankingTestCase(
        name="RAND Corporation",
        tier="tier_6_research_labs",
        anchor="RAND Corporation",
        positive="RAND Corporation",
        negatives=["UCLA", "USC", "Stanford"],
    ),
    RankingTestCase(
        name="Xerox PARC",
        tier="tier_6_research_labs",
        anchor="Palo Alto Research Center",
        positive="PARC",
        negatives=["Stanford", "UC Berkeley", "MIT"],
        notes="Formerly Xerox PARC",
    ),
    RankingTestCase(
        name="SRI International",
        tier="tier_6_research_labs",
        anchor="SRI International",
        positive="SRI International",
        negatives=["Stanford", "UC Berkeley", "MIT"],
    ),
    RankingTestCase(
        name="Battelle Memorial",
        tier="tier_6_research_labs",
        anchor="Battelle Memorial Institute",
        positive="Battelle",
        negatives=["Ohio State", "Case Western", "Oak Ridge"],
    ),
]

# TIER 7: Cross-lingual, transliteration (35 cases)

TIER_7_INTERNATIONAL = [
    RankingTestCase(
        name="Peking/Beijing",
        tier="tier_7_international",
        anchor="Peking University",
        positive="Beijing University",
        negatives=["Tsinghua University", "Beijing Normal University", "Renmin University"],
    ),
    RankingTestCase(
        name="PKU abbreviation",
        tier="tier_7_international",
        anchor="Peking University",
        positive="PKU",
        negatives=["Tsinghua", "Fudan", "Zhejiang University"],
    ),
    RankingTestCase(
        name="Tsinghua/Qinghua",
        tier="tier_7_international",
        anchor="Tsinghua University",
        positive="Qinghua University",
        negatives=["Peking University", "Fudan University", "Zhejiang University"],
    ),
    RankingTestCase(
        name="Fudan Chinese",
        tier="tier_7_international",
        anchor="Fudan University",
        positive="复旦大学",
        negatives=["Peking University", "Tsinghua University", "Shanghai Jiao Tong"],
        notes="Chinese characters",
    ),
    RankingTestCase(
        name="Tokyo/Todai",
        tier="tier_7_international",
        anchor="University of Tokyo",
        positive="Todai",
        negatives=["Kyoto University", "Osaka University", "Keio University"],
        notes="Japanese nickname",
    ),
    RankingTestCase(
        name="Kyoto Japanese",
        tier="tier_7_international",
        anchor="Kyoto University",
        positive="京都大学",
        negatives=["University of Tokyo", "Osaka University", "Tohoku University"],
    ),
    RankingTestCase(
        name="Seoul Korean",
        tier="tier_7_international",
        anchor="Seoul National University",
        positive="서울대학교",
        negatives=["Korea University", "Yonsei University", "KAIST"],
    ),
    RankingTestCase(
        name="Moscow Russian",
        tier="tier_7_international",
        anchor="Moscow State University",
        positive="МГУ",
        negatives=["St Petersburg University", "MIPT", "HSE Moscow"],
        notes="Russian abbreviation",
    ),
    RankingTestCase(
        name="Lomonosov full",
        tier="tier_7_international",
        anchor="Lomonosov Moscow State University",
        positive="Moscow State University",
        negatives=["St Petersburg University", "MIPT", "HSE Moscow"],
    ),
    RankingTestCase(
        name="ETH German full",
        tier="tier_7_international",
        anchor="ETH Zurich",
        positive="Eidgenössische Technische Hochschule Zürich",
        negatives=["EPFL", "University of Zurich", "University of Basel"],
    ),
    RankingTestCase(
        name="ETH English translation",
        tier="tier_7_international",
        anchor="ETH Zurich",
        positive="Swiss Federal Institute of Technology in Zurich",
        negatives=["EPFL", "University of Zurich", "University of Basel"],
    ),
    RankingTestCase(
        name="EPFL French full",
        tier="tier_7_international",
        anchor="EPFL",
        positive="École Polytechnique Fédérale de Lausanne",
        negatives=["ETH Zurich", "University of Geneva", "University of Lausanne"],
    ),
    RankingTestCase(
        name="Sorbonne evolution",
        tier="tier_7_international",
        anchor="Sorbonne University",
        positive="Sorbonne Université",
        negatives=["Paris-Saclay", "ENS Paris", "École Polytechnique"],
        notes="Post-2018 merger name",
    ),
    RankingTestCase(
        name="Paris-Saclay French",
        tier="tier_7_international",
        anchor="Paris-Saclay University",
        positive="Université Paris-Saclay",
        negatives=["Sorbonne", "ENS Paris", "École Polytechnique"],
    ),
    RankingTestCase(
        name="LMU German full",
        tier="tier_7_international",
        anchor="LMU Munich",
        positive="Ludwig-Maximilians-Universität München",
        negatives=["TU Munich", "University of Augsburg", "RWTH Aachen"],
    ),
    RankingTestCase(
        name="Humboldt German",
        tier="tier_7_international",
        anchor="Humboldt University of Berlin",
        positive="Humboldt-Universität zu Berlin",
        negatives=["Free University of Berlin", "TU Berlin", "University of Potsdam"],
    ),
    RankingTestCase(
        name="Rome Italian",
        tier="tier_7_international",
        anchor="Sapienza University of Rome",
        positive="Università degli Studi di Roma La Sapienza",
        negatives=["University of Bologna", "University of Milan", "Bocconi University"],
    ),
    RankingTestCase(
        name="Bologna Italian",
        tier="tier_7_international",
        anchor="University of Bologna",
        positive="Alma Mater Studiorum - Università di Bologna",
        negatives=["Sapienza Rome", "University of Milan", "University of Florence"],
    ),
    RankingTestCase(
        name="Leiden Dutch",
        tier="tier_7_international",
        anchor="Leiden University",
        positive="Universiteit Leiden",
        negatives=["TU Delft", "Utrecht University", "University of Amsterdam"],
    ),
    RankingTestCase(
        name="TU Delft Dutch",
        tier="tier_7_international",
        anchor="Delft University of Technology",
        positive="Technische Universiteit Delft",
        negatives=["TU Eindhoven", "Leiden University", "University of Amsterdam"],
    ),
    RankingTestCase(
        name="Uppsala Swedish",
        tier="tier_7_international",
        anchor="Uppsala University",
        positive="Uppsala universitet",
        negatives=["Lund University", "Stockholm University", "KTH"],
    ),
    RankingTestCase(
        name="Hebrew University",
        tier="tier_7_international",
        anchor="Hebrew University of Jerusalem",
        positive="האוניברסיטה העברית בירושלים",
        negatives=["Tel Aviv University", "Technion", "Weizmann Institute"],
        notes="Hebrew text",
    ),
    RankingTestCase(
        name="Technion Hebrew",
        tier="tier_7_international",
        anchor="Technion - Israel Institute of Technology",
        positive="הטכניון - מכון טכנולוגי לישראל",
        negatives=["Tel Aviv University", "Hebrew University", "Weizmann Institute"],
    ),
    RankingTestCase(
        name="National Taiwan",
        tier="tier_7_international",
        anchor="National Taiwan University",
        positive="國立臺灣大學",
        negatives=["National Tsing Hua", "National Chiao Tung", "NTNU Taiwan"],
        notes="Traditional Chinese",
    ),
    RankingTestCase(
        name="UNAM Spanish",
        tier="tier_7_international",
        anchor="National Autonomous University of Mexico",
        positive="Universidad Nacional Autónoma de México",
        negatives=["Tecnológico de Monterrey", "ITAM", "Universidad de Guadalajara"],
    ),
    RankingTestCase(
        name="USP Portuguese",
        tier="tier_7_international",
        anchor="University of São Paulo",
        positive="Universidade de São Paulo",
        negatives=["UNICAMP", "UFRJ", "UNESP"],
    ),
    RankingTestCase(
        name="São Paulo abbrev",
        tier="tier_7_international",
        anchor="University of São Paulo",
        positive="USP",
        negatives=["UNICAMP", "UFRJ", "UNESP"],
    ),
    RankingTestCase(
        name="Cairo Arabic",
        tier="tier_7_international",
        anchor="Cairo University",
        positive="جامعة القاهرة",
        negatives=["American University in Cairo", "Ain Shams University", "Alexandria University"],
        notes="Arabic text",
    ),
    RankingTestCase(
        name="IIT various",
        tier="tier_7_international",
        anchor="Indian Institute of Technology Bombay",
        positive="IIT Bombay",
        negatives=["IIT Delhi", "IIT Madras", "IIT Kanpur"],
    ),
    RankingTestCase(
        name="IIT Delhi Hindi",
        tier="tier_7_international",
        anchor="IIT Delhi",
        positive="भारतीय प्रौद्योगिकी संस्थान दिल्ली",
        negatives=["IIT Bombay", "IIT Madras", "IIT Kanpur"],
        notes="Hindi text",
    ),
    RankingTestCase(
        name="NTU Taiwan vs Singapore",
        tier="tier_7_international",
        anchor="National Taiwan University",
        positive="NTU Taiwan",
        negatives=["Nanyang Technological University", "NUS", "National Tsing Hua"],
        notes="NTU ambiguity",
    ),
    RankingTestCase(
        name="Nanyang as NTU",
        tier="tier_7_international",
        anchor="Nanyang Technological University",
        positive="NTU Singapore",
        negatives=["National Taiwan University", "NUS", "HKUST"],
    ),
    RankingTestCase(
        name="Waseda Japanese",
        tier="tier_7_international",
        anchor="Waseda University",
        positive="早稲田大学",
        negatives=["Keio University", "University of Tokyo", "Sophia University"],
    ),
    RankingTestCase(
        name="Keio Japanese",
        tier="tier_7_international",
        anchor="Keio University",
        positive="慶應義塾大学",
        negatives=["Waseda University", "University of Tokyo", "Sophia University"],
    ),
    RankingTestCase(
        name="Complutense Spanish",
        tier="tier_7_international",
        anchor="Complutense University of Madrid",
        positive="Universidad Complutense de Madrid",
        negatives=["Autonomous University of Madrid", "Carlos III University", "Polytechnic University of Madrid"],
    ),
    # =========================================================================
    # EXTENDED NON-LATIN CHARACTER TESTS
    # =========================================================================
    # ROMANIZATION VARIATIONS - Different transliteration systems
    RankingTestCase(
        name="Beijing Wade-Giles",
        tier="tier_7_international",
        anchor="北京大学",
        positive="Peking University",
        negatives=["Beijing Normal University", "Tsinghua University", "Renmin University"],
        notes="Wade-Giles romanization (Peking) vs Chinese characters",
    ),
    RankingTestCase(
        name="Beijing Pinyin",
        tier="tier_7_international",
        anchor="北京大学",
        positive="Beijing Daxue",
        negatives=["Beijing Normal University", "Tsinghua University", "Renmin University"],
        notes="Pinyin with tone-less transliteration",
    ),
    RankingTestCase(
        name="Tsinghua Wade-Giles vs Pinyin",
        tier="tier_7_international",
        anchor="清华大学",
        positive="Tsinghua University",
        negatives=["Peking University", "Fudan University", "Zhejiang University"],
        notes="Traditional Wade-Giles (Tsinghua) vs Simplified Chinese",
    ),
    RankingTestCase(
        name="Qinghua Pinyin variant",
        tier="tier_7_international",
        anchor="清华大学",
        positive="Qinghua Daxue",
        negatives=["Peking University", "Fudan University", "Zhejiang University"],
        notes="Pinyin romanization variant",
    ),
    RankingTestCase(
        name="Seoul McCune-Reischauer",
        tier="tier_7_international",
        anchor="서울대학교",
        positive="Sŏul Taehakkyo",
        negatives=["Korea University", "Yonsei University", "KAIST"],
        notes="McCune-Reischauer romanization with diacritics",
    ),
    RankingTestCase(
        name="Seoul Revised Romanization",
        tier="tier_7_international",
        anchor="서울대학교",
        positive="Seoul Daehakgyo",
        negatives=["Korea University", "Yonsei University", "KAIST"],
        notes="Revised Romanization of Korean",
    ),
    RankingTestCase(
        name="Tokyo Hepburn",
        tier="tier_7_international",
        anchor="東京大学",
        positive="Tōkyō Daigaku",
        negatives=["Kyoto University", "Osaka University", "Tohoku University"],
        notes="Hepburn romanization with macrons",
    ),
    RankingTestCase(
        name="Tokyo simplified",
        tier="tier_7_international",
        anchor="東京大学",
        positive="Tokyo Daigaku",
        negatives=["Kyoto University", "Osaka University", "Tohoku University"],
        notes="Simplified romanization without macrons",
    ),
    RankingTestCase(
        name="Moscow full Cyrillic",
        tier="tier_7_international",
        anchor="Московский государственный университет",
        positive="Moscow State University",
        negatives=["St Petersburg University", "MIPT", "HSE Moscow"],
        notes="Full Cyrillic to English",
    ),
    RankingTestCase(
        name="Moscow Lomonosov Cyrillic",
        tier="tier_7_international",
        anchor="Московский государственный университет имени М. В. Ломоносова",
        positive="Lomonosov Moscow State University",
        negatives=["St Petersburg University", "MIPT", "HSE Moscow"],
        notes="Full official Cyrillic name with Lomonosov",
    ),
    # DIACRITICS STRIPPING - OCR/noise scenarios where diacritics are lost
    RankingTestCase(
        name="Zurich no umlaut",
        tier="tier_7_international",
        anchor="ETH Zurich",
        positive="ETH Zürich",
        negatives=["EPFL", "University of Zurich", "University of Basel"],
        notes="Missing umlaut in anchor",
    ),
    RankingTestCase(
        name="Munchen no umlaut",
        tier="tier_7_international",
        anchor="Ludwig-Maximilians-Universitat Munchen",
        positive="Ludwig-Maximilians-Universität München",
        negatives=["TU Munich", "University of Augsburg", "RWTH Aachen"],
        notes="Multiple missing umlauts",
    ),
    RankingTestCase(
        name="Ecole no accent",
        tier="tier_7_international",
        anchor="Ecole Polytechnique Federale de Lausanne",
        positive="École Polytechnique Fédérale de Lausanne",
        negatives=["ETH Zurich", "University of Geneva", "University of Lausanne"],
        notes="Missing French accents",
    ),
    RankingTestCase(
        name="Autonoma no accent",
        tier="tier_7_international",
        anchor="Universidad Autonoma de Mexico",
        positive="Universidad Autónoma de México",
        negatives=["Tecnológico de Monterrey", "ITAM", "Universidad de Guadalajara"],
        notes="Missing Spanish accents",
    ),
    RankingTestCase(
        name="Sao Paulo no tilde",
        tier="tier_7_international",
        anchor="Universidade de Sao Paulo",
        positive="Universidade de São Paulo",
        negatives=["UNICAMP", "UFRJ", "UNESP"],
        notes="Missing Portuguese tilde",
    ),
    RankingTestCase(
        name="Goteborg Swedish",
        tier="tier_7_international",
        anchor="University of Goteborg",
        positive="Göteborgs universitet",
        negatives=["Uppsala University", "Lund University", "Stockholm University"],
        notes="Missing Swedish ö",
    ),
    RankingTestCase(
        name="Aarhus Danish",
        tier="tier_7_international",
        anchor="Aarhus University",
        positive="Aarhus Universitet",
        negatives=["Copenhagen University", "DTU", "Aalborg University"],
        notes="Danish å often written as aa",
    ),
    RankingTestCase(
        name="Lodz Polish",
        tier="tier_7_international",
        anchor="University of Lodz",
        positive="Uniwersytet Łódzki",
        negatives=["University of Warsaw", "Jagiellonian University", "AGH Krakow"],
        notes="Polish ł and ó stripped",
    ),
    RankingTestCase(
        name="Istanbul Turkish",
        tier="tier_7_international",
        anchor="Istanbul Technical University",
        positive="İstanbul Teknik Üniversitesi",
        negatives=["Boğaziçi University", "METU", "Koç University"],
        notes="Turkish İ and ü",
    ),
    RankingTestCase(
        name="Malaga Spanish",
        tier="tier_7_international",
        anchor="Universidad de Malaga",
        positive="Universidad de Málaga",
        negatives=["University of Seville", "University of Granada", "University of Valencia"],
        notes="Missing Spanish acute accent",
    ),
    # MIXED SCRIPT - Partial transliteration/mixed languages
    RankingTestCase(
        name="Peking mixed",
        tier="tier_7_international",
        anchor="Peking 大学",
        positive="Peking University",
        negatives=["Tsinghua University", "Beijing Normal University", "Renmin University"],
        notes="Mixed English and Chinese characters",
    ),
    RankingTestCase(
        name="Tokyo mixed",
        tier="tier_7_international",
        anchor="Tokyo 大学",
        positive="University of Tokyo",
        negatives=["Kyoto University", "Osaka University", "Tohoku University"],
        notes="Mixed English city name with Japanese suffix",
    ),
    RankingTestCase(
        name="Seoul mixed",
        tier="tier_7_international",
        anchor="Seoul 대학교",
        positive="Seoul National University",
        negatives=["Korea University", "Yonsei University", "KAIST"],
        notes="Mixed English city with Korean suffix",
    ),
    RankingTestCase(
        name="Moscow mixed",
        tier="tier_7_international",
        anchor="Moscow университет",
        positive="Moscow State University",
        negatives=["St Petersburg University", "MIPT", "HSE Moscow"],
        notes="Mixed English and Cyrillic",
    ),
    RankingTestCase(
        name="Cairo mixed",
        tier="tier_7_international",
        anchor="Cairo جامعة",
        positive="Cairo University",
        negatives=["American University in Cairo", "Ain Shams University", "Alexandria University"],
        notes="Mixed English and Arabic (جامعة = university)",
    ),
    RankingTestCase(
        name="Tel Aviv mixed",
        tier="tier_7_international",
        anchor="Tel Aviv אוניברסיטת",
        positive="Tel Aviv University",
        negatives=["Hebrew University", "Technion", "Weizmann Institute"],
        notes="Mixed English and Hebrew",
    ),
    # UNICODE NORMALIZATION - Full-width, combining characters, etc.
    RankingTestCase(
        name="Tokyo fullwidth",
        tier="tier_7_international",
        anchor="Ｔｏｋｙｏ　Ｕｎｉｖｅｒｓｉｔｙ",
        positive="University of Tokyo",
        negatives=["Kyoto University", "Osaka University", "Tohoku University"],
        notes="Full-width Latin characters (common in Japanese documents)",
    ),
    RankingTestCase(
        name="Seoul fullwidth",
        tier="tier_7_international",
        anchor="Ｓｅｏｕｌ　Ｎａｔｉｏｎａｌ　Ｕｎｉｖｅｒｓｉｔｙ",
        positive="Seoul National University",
        negatives=["Korea University", "Yonsei University", "KAIST"],
        notes="Full-width Latin (common in Korean documents)",
    ),
    RankingTestCase(
        name="Zurich combining",
        tier="tier_7_international",
        anchor="ETH Zu\u0308rich",
        positive="ETH Zürich",
        negatives=["EPFL", "University of Zurich", "University of Basel"],
        notes="NFD form with combining diaeresis (u + ̈)",
    ),
    RankingTestCase(
        name="Ecole combining",
        tier="tier_7_international",
        anchor="E\u0301cole Polytechnique",
        positive="École Polytechnique",
        negatives=["ENS Paris", "Paris-Saclay", "Sorbonne University"],
        notes="NFD form with combining acute accent (E + ́)",
    ),
    RankingTestCase(
        name="Sao combining",
        tier="tier_7_international",
        anchor="Universidade de Sa\u0303o Paulo",
        positive="Universidade de São Paulo",
        negatives=["UNICAMP", "UFRJ", "UNESP"],
        notes="NFD form with combining tilde (a + ̃)",
    ),
    RankingTestCase(
        name="Harvard zero-width",
        tier="tier_7_international",
        anchor="Harvard\u200BUniversity",
        positive="Harvard University",
        negatives=["Yale University", "Princeton University", "Columbia University"],
        notes="Zero-width space (U+200B) between words",
    ),
    RankingTestCase(
        name="MIT soft hyphen",
        tier="tier_7_international",
        anchor="Massa\u00ADchusetts Institute of Technology",
        positive="Massachusetts Institute of Technology",
        negatives=["Stanford", "Caltech", "Georgia Tech"],
        notes="Soft hyphen (U+00AD) in the middle of word",
    ),
    RankingTestCase(
        name="Stanford NBSP",
        tier="tier_7_international",
        anchor="Stanford\u00A0University",
        positive="Stanford University",
        negatives=["UC Berkeley", "USC", "UCLA"],
        notes="Non-breaking space (U+00A0) instead of regular space",
    ),
    # ADDITIONAL SCRIPTS - Greek, Thai, more Cyrillic, Vietnamese
    RankingTestCase(
        name="Athens Greek",
        tier="tier_7_international",
        anchor="National and Kapodistrian University of Athens",
        positive="Εθνικό και Καποδιστριακό Πανεπιστήμιο Αθηνών",
        negatives=["Aristotle University of Thessaloniki", "University of Patras", "Athens University of Economics"],
        notes="Greek script",
    ),
    RankingTestCase(
        name="Thessaloniki Greek",
        tier="tier_7_international",
        anchor="Aristotle University of Thessaloniki",
        positive="Αριστοτέλειο Πανεπιστήμιο Θεσσαλονίκης",
        negatives=["University of Athens", "University of Patras", "University of Crete"],
        notes="Greek script with English name",
    ),
    RankingTestCase(
        name="Chulalongkorn Thai",
        tier="tier_7_international",
        anchor="Chulalongkorn University",
        positive="จุฬาลงกรณ์มหาวิทยาลัย",
        negatives=["Mahidol University", "Kasetsart University", "Thammasat University"],
        notes="Thai script",
    ),
    RankingTestCase(
        name="Mahidol Thai",
        tier="tier_7_international",
        anchor="Mahidol University",
        positive="มหาวิทยาลัยมหิดล",
        negatives=["Chulalongkorn University", "Kasetsart University", "Thammasat University"],
        notes="Thai script",
    ),
    RankingTestCase(
        name="Hanoi Vietnamese",
        tier="tier_7_international",
        anchor="Vietnam National University, Hanoi",
        positive="Đại học Quốc gia Hà Nội",
        negatives=["VNU Ho Chi Minh City", "Hanoi University of Science and Technology", "Foreign Trade University"],
        notes="Vietnamese with diacritics",
    ),
    RankingTestCase(
        name="Ho Chi Minh Vietnamese",
        tier="tier_7_international",
        anchor="Vietnam National University Ho Chi Minh City",
        positive="Đại học Quốc gia Thành phố Hồ Chí Minh",
        negatives=["VNU Hanoi", "Ho Chi Minh City University of Technology", "University of Economics HCMC"],
        notes="Vietnamese Ho Chi Minh vs English",
    ),
    RankingTestCase(
        name="Kyiv Ukrainian",
        tier="tier_7_international",
        anchor="Taras Shevchenko National University of Kyiv",
        positive="Київський національний університет імені Тараса Шевченка",
        negatives=["Igor Sikorsky Kyiv Polytechnic Institute", "Kharkiv National University", "Lviv University"],
        notes="Ukrainian Cyrillic",
    ),
    RankingTestCase(
        name="Kyiv vs Kiev spelling",
        tier="tier_7_international",
        anchor="Київський національний університет",
        positive="Kyiv National University",
        negatives=["Kiev Polytechnic", "Kharkiv University", "Lviv University"],
        notes="Kyiv (Ukrainian) vs Kiev (Russian) romanization",
    ),
    RankingTestCase(
        name="Prague Czech",
        tier="tier_7_international",
        anchor="Charles University",
        positive="Univerzita Karlova",
        negatives=["Czech Technical University", "Masaryk University", "Palacký University"],
        notes="Czech name for Charles University in Prague",
    ),
    RankingTestCase(
        name="Warsaw Polish",
        tier="tier_7_international",
        anchor="University of Warsaw",
        positive="Uniwersytet Warszawski",
        negatives=["Warsaw University of Technology", "Jagiellonian University", "University of Wrocław"],
        notes="Polish name",
    ),
    RankingTestCase(
        name="Krakow Jagiellonian",
        tier="tier_7_international",
        anchor="Jagiellonian University",
        positive="Uniwersytet Jagielloński w Krakowie",
        negatives=["University of Warsaw", "AGH Krakow", "University of Wrocław"],
        notes="Polish name with w Krakowie suffix",
    ),
    RankingTestCase(
        name="Budapest Hungarian",
        tier="tier_7_international",
        anchor="Eötvös Loránd University",
        positive="Eötvös Loránd Tudományegyetem",
        negatives=["Budapest University of Technology", "Semmelweis University", "Corvinus University"],
        notes="Hungarian name with special characters ö, á",
    ),
    RankingTestCase(
        name="Helsinki Finnish",
        tier="tier_7_international",
        anchor="University of Helsinki",
        positive="Helsingin yliopisto",
        negatives=["Aalto University", "University of Turku", "Tampere University"],
        notes="Finnish name",
    ),
    RankingTestCase(
        name="Zhejiang simplified",
        tier="tier_7_international",
        anchor="浙江大学",
        positive="Zhejiang University",
        negatives=["Fudan University", "Nanjing University", "Shanghai Jiao Tong University"],
        notes="Simplified Chinese to English",
    ),
    RankingTestCase(
        name="Nanjing Chinese",
        tier="tier_7_international",
        anchor="南京大学",
        positive="Nanjing University",
        negatives=["Southeast University", "Hohai University", "Nanjing Normal University"],
        notes="Chinese characters (Simplified)",
    ),
    RankingTestCase(
        name="Hong Kong traditional",
        tier="tier_7_international",
        anchor="香港大學",
        positive="University of Hong Kong",
        negatives=["Chinese University of Hong Kong", "HKUST", "City University of Hong Kong"],
        notes="Traditional Chinese characters",
    ),
    RankingTestCase(
        name="CUHK Chinese",
        tier="tier_7_international",
        anchor="香港中文大學",
        positive="Chinese University of Hong Kong",
        negatives=["University of Hong Kong", "HKUST", "City University of Hong Kong"],
        notes="Traditional Chinese for CUHK",
    ),
    RankingTestCase(
        name="NUS Chinese",
        tier="tier_7_international",
        anchor="新加坡国立大学",
        positive="National University of Singapore",
        negatives=["Nanyang Technological University", "Singapore Management University", "SUTD"],
        notes="Simplified Chinese for NUS",
    ),
    RankingTestCase(
        name="Osaka Japanese",
        tier="tier_7_international",
        anchor="大阪大学",
        positive="Osaka University",
        negatives=["University of Tokyo", "Kyoto University", "Nagoya University"],
        notes="Japanese kanji",
    ),
    RankingTestCase(
        name="Tohoku Japanese",
        tier="tier_7_international",
        anchor="東北大学",
        positive="Tohoku University",
        negatives=["University of Tokyo", "Hokkaido University", "Kyushu University"],
        notes="Japanese kanji for Tohoku",
    ),
    RankingTestCase(
        name="KAIST Korean",
        tier="tier_7_international",
        anchor="한국과학기술원",
        positive="Korea Advanced Institute of Science and Technology",
        negatives=["Seoul National University", "POSTECH", "Korea University"],
        notes="Korean to full English name",
    ),
    RankingTestCase(
        name="Yonsei Korean",
        tier="tier_7_international",
        anchor="연세대학교",
        positive="Yonsei University",
        negatives=["Seoul National University", "Korea University", "KAIST"],
        notes="Korean script to English",
    ),
    RankingTestCase(
        name="St Petersburg Cyrillic",
        tier="tier_7_international",
        anchor="Санкт-Петербургский государственный университет",
        positive="Saint Petersburg State University",
        negatives=["Moscow State University", "ITMO University", "HSE St Petersburg"],
        notes="Full Cyrillic name",
    ),
    RankingTestCase(
        name="Novosibirsk Cyrillic",
        tier="tier_7_international",
        anchor="Новосибирский государственный университет",
        positive="Novosibirsk State University",
        negatives=["Tomsk State University", "Ural Federal University", "Siberian Federal University"],
        notes="Siberian university in Cyrillic",
    ),
    RankingTestCase(
        name="Al-Azhar Arabic",
        tier="tier_7_international",
        anchor="جامعة الأزهر",
        positive="Al-Azhar University",
        negatives=["Cairo University", "Alexandria University", "Ain Shams University"],
        notes="Arabic script for Al-Azhar",
    ),
    RankingTestCase(
        name="King Saud Arabic",
        tier="tier_7_international",
        anchor="جامعة الملك سعود",
        positive="King Saud University",
        negatives=["King Abdulaziz University", "KAUST", "King Fahd University"],
        notes="Arabic script for Saudi university",
    ),
    RankingTestCase(
        name="Tehran Persian",
        tier="tier_7_international",
        anchor="دانشگاه تهران",
        positive="University of Tehran",
        negatives=["Sharif University", "Amirkabir University", "Iran University of Science and Technology"],
        notes="Persian/Farsi script",
    ),
    RankingTestCase(
        name="IIT Bombay Devanagari",
        tier="tier_7_international",
        anchor="भारतीय प्रौद्योगिकी संस्थान मुंबई",
        positive="Indian Institute of Technology Bombay",
        negatives=["IIT Delhi", "IIT Madras", "IIT Kanpur"],
        notes="Hindi/Devanagari for IIT Bombay",
    ),
    RankingTestCase(
        name="IISc Bangalore Devanagari",
        tier="tier_7_international",
        anchor="भारतीय विज्ञान संस्थान",
        positive="Indian Institute of Science",
        negatives=["IIT Bangalore", "IIIT Bangalore", "Bangalore University"],
        notes="Hindi name for IISc",
    ),
    RankingTestCase(
        name="Bangladesh Bangla",
        tier="tier_7_international",
        anchor="ঢাকা বিশ্ববিদ্যালয়",
        positive="University of Dhaka",
        negatives=["Bangladesh University of Engineering", "Jahangirnagar University", "Chittagong University"],
        notes="Bengali/Bangla script",
    ),
    RankingTestCase(
        name="Colombo Sinhala",
        tier="tier_7_international",
        anchor="කොළඹ විශ්වවිද්‍යාලය",
        positive="University of Colombo",
        negatives=["University of Peradeniya", "University of Kelaniya", "University of Moratuwa"],
        notes="Sinhala script (Sri Lanka)",
    ),
    # EDGE CASES - Unusual but realistic scenarios
    RankingTestCase(
        name="CJK spaces",
        tier="tier_7_international",
        anchor="北京　大学",
        positive="Peking University",
        negatives=["Tsinghua University", "Beijing Normal University", "Renmin University"],
        notes="Ideographic space (U+3000) between characters",
    ),
    RankingTestCase(
        name="Arabic-Indic numerals",
        tier="tier_7_international",
        anchor="جامعة الملك عبدالله للعلوم والتقنية",
        positive="King Abdullah University of Science and Technology",
        negatives=["King Saud University", "King Fahd University", "Taibah University"],
        notes="KAUST in Arabic script",
    ),
    RankingTestCase(
        name="Devanagari numerals",
        tier="tier_7_international",
        anchor="दिल्ली विश्वविद्यालय",
        positive="University of Delhi",
        negatives=["JNU Delhi", "IIT Delhi", "Delhi Technological University"],
        notes="Hindi name for University of Delhi",
    ),
]

# TIER 8: DISAMBIGUATION (30 cases) - Confusable institutions

TIER_8_DISAMBIGUATION = [
    RankingTestCase(
        name="Washington vs Washington St. Louis",
        tier="tier_8_disambiguation",
        anchor="University of Washington",
        positive="UW Seattle",
        negatives=["Washington University in St. Louis", "George Washington University", "Washington State University"],
    ),
    RankingTestCase(
        name="Wash U St. Louis",
        tier="tier_8_disambiguation",
        anchor="Washington University in St. Louis",
        positive="WashU",
        negatives=["University of Washington", "George Washington University", "Washington State University"],
    ),
    RankingTestCase(
        name="George Washington",
        tier="tier_8_disambiguation",
        anchor="George Washington University",
        positive="GW",
        negatives=["University of Washington", "Washington University in St. Louis", "Georgetown"],
    ),
    RankingTestCase(
        name="Cambridge UK vs MIT",
        tier="tier_8_disambiguation",
        anchor="University of Cambridge",
        positive="Cambridge UK",
        negatives=["MIT", "Harvard University", "Boston University"],
        notes="Cambridge, MA confusion",
    ),
    RankingTestCase(
        name="Georgia Tech vs Georgia State",
        tier="tier_8_disambiguation",
        anchor="Georgia Institute of Technology",
        positive="Georgia Tech",
        negatives=["Georgia State University", "University of Georgia", "Emory University"],
    ),
    RankingTestCase(
        name="Georgia State",
        tier="tier_8_disambiguation",
        anchor="Georgia State University",
        positive="GSU",
        negatives=["Georgia Tech", "University of Georgia", "Emory University"],
    ),
    RankingTestCase(
        name="UGA vs Georgia Tech",
        tier="tier_8_disambiguation",
        anchor="University of Georgia",
        positive="UGA",
        negatives=["Georgia Tech", "Georgia State University", "Emory University"],
    ),
    RankingTestCase(
        name="Penn vs Penn State",
        tier="tier_8_disambiguation",
        anchor="University of Pennsylvania",
        positive="Penn",
        negatives=["Penn State University", "Temple University", "Drexel University"],
    ),
    RankingTestCase(
        name="Penn State proper",
        tier="tier_8_disambiguation",
        anchor="Pennsylvania State University",
        positive="Penn State",
        negatives=["University of Pennsylvania", "Temple University", "Carnegie Mellon"],
    ),
    RankingTestCase(
        name="USC vs South Carolina",
        tier="tier_8_disambiguation",
        anchor="University of Southern California",
        positive="USC Trojans",
        negatives=["University of South Carolina", "UCLA", "Stanford"],
        notes="USC abbreviation ambiguity",
    ),
    RankingTestCase(
        name="Miami FL vs Miami OH",
        tier="tier_8_disambiguation",
        anchor="University of Miami",
        positive="Miami FL",
        negatives=["Miami University (Ohio)", "Florida State", "Florida International"],
    ),
    RankingTestCase(
        name="Miami of Ohio",
        tier="tier_8_disambiguation",
        anchor="Miami University",
        positive="Miami OH",
        negatives=["University of Miami", "Ohio State", "Ohio University"],
        notes="Miami University is in Ohio",
    ),
    RankingTestCase(
        name="Ohio vs Ohio State",
        tier="tier_8_disambiguation",
        anchor="Ohio University",
        positive="OU Athens",
        negatives=["Ohio State University", "Kent State", "Miami University"],
    ),
    RankingTestCase(
        name="Ohio State proper",
        tier="tier_8_disambiguation",
        anchor="Ohio State University",
        positive="OSU Columbus",
        negatives=["Ohio University", "Oklahoma State", "Oregon State"],
    ),
    RankingTestCase(
        name="Boston U vs Boston College",
        tier="tier_8_disambiguation",
        anchor="Boston University",
        positive="BU",
        negatives=["Boston College", "Northeastern University", "Tufts University"],
    ),
    RankingTestCase(
        name="Boston College proper",
        tier="tier_8_disambiguation",
        anchor="Boston College",
        positive="BC",
        negatives=["Boston University", "Northeastern University", "Tufts University"],
    ),
    RankingTestCase(
        name="Indiana vs IU Bloomington",
        tier="tier_8_disambiguation",
        anchor="Indiana University Bloomington",
        positive="IU Bloomington",
        negatives=["Purdue University", "Indiana State", "Ball State"],
    ),
    RankingTestCase(
        name="Illinois vs UIC",
        tier="tier_8_disambiguation",
        anchor="University of Illinois at Urbana-Champaign",
        positive="UIUC",
        negatives=["UIC", "Illinois State", "Northwestern"],
    ),
    RankingTestCase(
        name="UIC proper",
        tier="tier_8_disambiguation",
        anchor="University of Illinois at Chicago",
        positive="UIC",
        negatives=["UIUC", "Illinois State", "DePaul"],
    ),
    RankingTestCase(
        name="Michigan vs Michigan State",
        tier="tier_8_disambiguation",
        anchor="University of Michigan",
        positive="UMich",
        negatives=["Michigan State University", "Wayne State", "Western Michigan"],
    ),
    RankingTestCase(
        name="MSU Michigan",
        tier="tier_8_disambiguation",
        anchor="Michigan State University",
        positive="MSU East Lansing",
        negatives=["University of Michigan", "Wayne State", "Mississippi State"],
        notes="MSU ambiguity",
    ),
    RankingTestCase(
        name="UC Berkeley vs UC Davis",
        tier="tier_8_disambiguation",
        anchor="University of California, Berkeley",
        positive="Berkeley",
        negatives=["UC Davis", "UCLA", "UCSD"],
    ),
    RankingTestCase(
        name="UC system disambiguation",
        tier="tier_8_disambiguation",
        anchor="University of California",
        positive="UC System",
        negatives=["USC", "Stanford", "Caltech"],
        notes="UC refers to system",
    ),
    RankingTestCase(
        name="UT Austin vs UT Dallas",
        tier="tier_8_disambiguation",
        anchor="University of Texas at Austin",
        positive="UT Austin",
        negatives=["UT Dallas", "UT San Antonio", "Texas A&M"],
    ),
    RankingTestCase(
        name="Imperial UK",
        tier="tier_8_disambiguation",
        anchor="Imperial College London",
        positive="Imperial",
        negatives=["UCL", "King's College London", "University of Manchester"],
    ),
    RankingTestCase(
        name="King's London vs Cambridge",
        tier="tier_8_disambiguation",
        anchor="King's College London",
        positive="KCL",
        negatives=["King's College Cambridge", "UCL", "LSE"],
    ),
    RankingTestCase(
        name="Trinity Dublin vs Cambridge",
        tier="tier_8_disambiguation",
        anchor="Trinity College Dublin",
        positive="TCD",
        negatives=["Trinity College Cambridge", "University College Dublin", "Dublin City University"],
    ),
    RankingTestCase(
        name="St Andrews Scotland",
        tier="tier_8_disambiguation",
        anchor="University of St Andrews",
        positive="St Andrews",
        negatives=["University of Edinburgh", "University of Glasgow", "University of Aberdeen"],
    ),
    RankingTestCase(
        name="Victoria multiple",
        tier="tier_8_disambiguation",
        anchor="University of Victoria",
        positive="UVic Canada",
        negatives=["Victoria University of Wellington", "University of Melbourne", "University of Toronto"],
        notes="Multiple Victoria universities",
    ),
    RankingTestCase(
        name="Newcastle UK vs Australia",
        tier="tier_8_disambiguation",
        anchor="Newcastle University",
        positive="Newcastle UK",
        negatives=["University of Newcastle Australia", "Durham University", "University of Leeds"],
    ),
]

# TIER 9: NEGATIVE CONTROLS (20 cases) - Should NOT match

TIER_9_NEGATIVE = [
    RankingTestCase(
        name="Harvard not Yale",
        tier="tier_9_negative",
        anchor="Harvard University",
        positive="Harvard University",
        negatives=["Yale University", "Princeton University", "Columbia University"],
        notes="Positive IS anchor; negatives must lose",
    ),
    RankingTestCase(
        name="MIT not Stanford",
        tier="tier_9_negative",
        anchor="Massachusetts Institute of Technology",
        positive="MIT",
        negatives=["Stanford University", "Caltech", "Carnegie Mellon"],
    ),
    RankingTestCase(
        name="Oxford not Cambridge",
        tier="tier_9_negative",
        anchor="University of Oxford",
        positive="Oxford University",
        negatives=["University of Cambridge", "Imperial College", "UCL"],
    ),
    RankingTestCase(
        name="Berkeley not UCLA",
        tier="tier_9_negative",
        anchor="University of California, Berkeley",
        positive="UC Berkeley",
        negatives=["UCLA", "UCSD", "Stanford"],
    ),
    RankingTestCase(
        name="Columbia not Cornell",
        tier="tier_9_negative",
        anchor="Columbia University",
        positive="Columbia",
        negatives=["Cornell University", "NYU", "Brown University"],
    ),
    RankingTestCase(
        name="Duke not UNC",
        tier="tier_9_negative",
        anchor="Duke University",
        positive="Duke",
        negatives=["UNC Chapel Hill", "NC State", "Wake Forest"],
    ),
    RankingTestCase(
        name="Northwestern not UChicago",
        tier="tier_9_negative",
        anchor="Northwestern University",
        positive="Northwestern",
        negatives=["University of Chicago", "Illinois", "Notre Dame"],
    ),
    RankingTestCase(
        name="Carnegie Mellon not Pitt",
        tier="tier_8_disambiguation",
        anchor="Carnegie Mellon University",
        positive="CMU",
        negatives=["University of Pittsburgh", "Penn State", "Duquesne"],
    ),
    RankingTestCase(
        name="Rice not Texas",
        tier="tier_9_negative",
        anchor="Rice University",
        positive="Rice",
        negatives=["University of Texas at Austin", "Texas A&M", "Baylor"],
    ),
    RankingTestCase(
        name="Vanderbilt not Tennessee",
        tier="tier_9_negative",
        anchor="Vanderbilt University",
        positive="Vanderbilt",
        negatives=["University of Tennessee", "Belmont University", "Lipscomb University"],
    ),
    RankingTestCase(
        name="Emory not Georgia",
        tier="tier_9_negative",
        anchor="Emory University",
        positive="Emory",
        negatives=["University of Georgia", "Georgia Tech", "Georgia State"],
    ),
    RankingTestCase(
        name="Tulane not LSU",
        tier="tier_9_negative",
        anchor="Tulane University",
        positive="Tulane",
        negatives=["Louisiana State University", "University of New Orleans", "Loyola New Orleans"],
    ),
    RankingTestCase(
        name="Notre Dame not Purdue",
        tier="tier_9_negative",
        anchor="University of Notre Dame",
        positive="Notre Dame",
        negatives=["Purdue University", "Indiana University", "Ball State"],
    ),
    RankingTestCase(
        name="Johns Hopkins not Maryland",
        tier="tier_9_negative",
        anchor="Johns Hopkins University",
        positive="Hopkins",
        negatives=["University of Maryland", "Georgetown", "Towson University"],
    ),
    RankingTestCase(
        name="Brown not RISD",
        tier="tier_9_negative",
        anchor="Brown University",
        positive="Brown",
        negatives=["Rhode Island School of Design", "Providence College", "Bryant University"],
    ),
    RankingTestCase(
        name="Dartmouth not Vermont",
        tier="tier_9_negative",
        anchor="Dartmouth College",
        positive="Dartmouth",
        negatives=["University of Vermont", "Middlebury College", "Williams College"],
    ),
    RankingTestCase(
        name="Acknowledgements trap",
        tier="tier_9_negative",
        anchor="Acknowledgements",
        positive="NOT_A_MATCH",
        negatives=["Harvard University", "MIT", "Stanford"],
        notes="Not an affiliation at all",
    ),
    RankingTestCase(
        name="Supplementary Materials trap",
        tier="tier_9_negative",
        anchor="Supplementary Materials",
        positive="NOT_A_MATCH",
        negatives=["Oxford University", "Cambridge University", "Imperial College"],
        notes="Section header, not affiliation",
    ),
    RankingTestCase(
        name="References section trap",
        tier="tier_9_negative",
        anchor="References",
        positive="NOT_A_MATCH",
        negatives=["Princeton University", "Yale University", "Columbia University"],
        notes="Section header",
    ),
    RankingTestCase(
        name="Author Contributions trap",
        tier="tier_9_negative",
        anchor="Author Contributions",
        positive="NOT_A_MATCH",
        negatives=["Stanford University", "MIT", "Caltech"],
        notes="Section header",
    ),
]

# TIER 10: ULTRA-HARD (30 cases) - Combined challenges

TIER_10_ULTRAHARD = [
    RankingTestCase(
        name="OCR + abbreviation MIT",
        tier="tier_10_ultrahard",
        anchor="Dept. Phys., M1T",
        positive="Massachusetts Institute of Technology",
        negatives=["Stanford", "Caltech", "Princeton"],
        notes="OCR 1 for I + department abbrev",
    ),
    RankingTestCase(
        name="Chinese + department",
        tier="tier_10_ultrahard",
        anchor="Dept. of CS, PKU, Beijing 100871",
        positive="Peking University",
        negatives=["Tsinghua University", "Fudan University", "Zhejiang University"],
        notes="Abbreviation + postal code",
    ),
    RankingTestCase(
        name="Hospital + OCR noise",
        tier="tier_10_ultrahard",
        anchor="Mass Gen Hosp., Harvard Med",
        positive="Harvard University",
        negatives=["Boston University", "Tufts", "MIT"],
        notes="Hospital abbrev + school abbrev",
    ),
    RankingTestCase(
        name="German + abbreviation + OCR",
        tier="tier_10_ultrahard",
        anchor="TU Munchen",
        positive="Technical University of Munich",
        negatives=["LMU Munich", "RWTH Aachen", "University of Stuttgart"],
        notes="Missing umlaut + abbreviation",
    ),
    RankingTestCase(
        name="Lab + parent + noise",
        tier="tier_10_ultrahard",
        anchor="MIT Lincoln Lab, Lexington MA",
        positive="Massachusetts Institute of Technology",
        negatives=["Harvard", "Boston University", "Northeastern"],
        notes="Lab + location",
    ),
    RankingTestCase(
        name="French + abbreviation",
        tier="tier_10_ultrahard",
        anchor="Ecole Polytechnique Fed. de Lausanne",
        positive="EPFL",
        negatives=["ETH Zurich", "University of Geneva", "University of Lausanne"],
        notes="Missing accents + truncation",
    ),
    RankingTestCase(
        name="Medical + disambiguation",
        tier="tier_10_ultrahard",
        anchor="Johns Hopkins Hosp., Baltimore MD",
        positive="Johns Hopkins University",
        negatives=["University of Maryland", "Georgetown", "George Washington"],
        notes="Hospital + city",
    ),
    RankingTestCase(
        name="Multi-affiliation",
        tier="tier_10_ultrahard",
        anchor="Stanford University; Google Research",
        positive="Stanford University",
        negatives=["MIT", "UC Berkeley", "CMU"],
        notes="Dual affiliation",
    ),
    RankingTestCase(
        name="Broad Institute joint",
        tier="tier_10_ultrahard",
        anchor="Broad Inst. of MIT & Harvard",
        positive="MIT",
        negatives=["Harvard University", "Boston University", "Northeastern"],
        notes="Joint institute abbreviated",
    ),
    RankingTestCase(
        name="National lab + noise",
        tier="tier_10_ultrahard",
        anchor="LBNL, Berkeley CA 94720",
        positive="Lawrence Berkeley National Laboratory",
        negatives=["Stanford", "SLAC", "Caltech"],
        notes="Abbreviation + zip code",
    ),
    RankingTestCase(
        name="All caps + department",
        tier="tier_10_ultrahard",
        anchor="DEPARTMENT OF PHYSICS, STANFORD UNIVERSITY",
        positive="Stanford University",
        negatives=["MIT", "Caltech", "Princeton"],
        notes="PDF header extraction",
    ),
    RankingTestCase(
        name="Japanese + romanization",
        tier="tier_10_ultrahard",
        anchor="Univ. of Tokyo, Dept. of Physics",
        positive="University of Tokyo",
        negatives=["Kyoto University", "Osaka University", "Tohoku University"],
        notes="Abbreviation + department",
    ),
    RankingTestCase(
        name="UK college + abbreviation",
        tier="tier_10_ultrahard",
        anchor="Trinity Coll., Univ. of Cambridge",
        positive="University of Cambridge",
        negatives=["Oxford University", "Imperial College", "UCL"],
        notes="College + abbreviations",
    ),
    RankingTestCase(
        name="Medical school + city confusion",
        tier="tier_10_ultrahard",
        anchor="Columbia Med School, NYC",
        positive="Columbia University",
        negatives=["NYU", "Cornell Weill", "Mount Sinai"],
        notes="Abbreviated medical school",
    ),
    RankingTestCase(
        name="Corporate + academic",
        tier="tier_10_ultrahard",
        anchor="Google Brain / Stanford CS",
        positive="Stanford University",
        negatives=["MIT", "CMU", "UC Berkeley"],
        notes="Industry-academia joint",
    ),
    RankingTestCase(
        name="Address style affiliation",
        tier="tier_10_ultrahard",
        anchor="77 Massachusetts Ave, Cambridge, MA 02139",
        positive="Massachusetts Institute of Technology",
        negatives=["Harvard University", "Boston University", "Northeastern"],
        notes="Address instead of name",
    ),
    RankingTestCase(
        name="Merged text + hospital",
        tier="tier_10_ultrahard",
        anchor="DanaFarberCancerInstitute",
        positive="Harvard University",
        negatives=["Boston University", "Tufts", "MIT"],
        notes="OCR merged words",
    ),
    RankingTestCase(
        name="Korean + abbreviation",
        tier="tier_10_ultrahard",
        anchor="Dept. of EE, SNU, Seoul",
        positive="Seoul National University",
        negatives=["KAIST", "Korea University", "Yonsei University"],
        notes="Abbreviation + department + city",
    ),
    RankingTestCase(
        name="German institute + noise",
        tier="tier_10_ultrahard",
        anchor="Max Planck Inst. Astrophys., Garching",
        positive="Max Planck Institute for Astrophysics",
        negatives=["ESO", "CERN", "DLR"],
        notes="Abbreviation + location",
    ),
    RankingTestCase(
        name="Spanish + abbreviation",
        tier="tier_10_ultrahard",
        anchor="UNAM, Mexico City",
        positive="National Autonomous University of Mexico",
        negatives=["Tecnológico de Monterrey", "ITAM", "Universidad de Guadalajara"],
        notes="Abbreviation + city",
    ),
    RankingTestCase(
        name="Consortium affiliation",
        tier="tier_10_ultrahard",
        anchor="LIGO Scientific Collaboration, Caltech",
        positive="California Institute of Technology",
        negatives=["MIT", "Stanford", "UC Berkeley"],
        notes="Collaboration + institution",
    ),
    RankingTestCase(
        name="Historical name + current",
        tier="tier_10_ultrahard",
        anchor="Paris VI (now Sorbonne Université)",
        positive="Sorbonne University",
        negatives=["ENS Paris", "École Polytechnique", "Paris-Saclay"],
        notes="Historical with update",
    ),
    RankingTestCase(
        name="Research division + parent",
        tier="tier_10_ultrahard",
        anchor="IBM Research - Yorktown Heights",
        positive="IBM Research",
        negatives=["MIT", "Stanford", "CMU"],
        notes="Corporate research site",
    ),
    RankingTestCase(
        name="Multiple typos",
        tier="tier_10_ultrahard",
        anchor="Massechusetts lnstitute of Tecnology",
        positive="Massachusetts Institute of Technology",
        negatives=["Stanford", "Caltech", "Georgia Tech"],
        notes="Multiple OCR errors",
    ),
    RankingTestCase(
        name="Chinese city + university",
        tier="tier_10_ultrahard",
        anchor="Tsinghua Univ., Beijing, China",
        positive="Tsinghua University",
        negatives=["Peking University", "Fudan University", "Zhejiang University"],
        notes="Abbreviation + location",
    ),
    RankingTestCase(
        name="Email domain extraction",
        tier="tier_10_ultrahard",
        anchor="john.doe@cs.stanford.edu",
        positive="Stanford University",
        negatives=["MIT", "UC Berkeley", "CMU"],
        notes="Email as affiliation",
    ),
    RankingTestCase(
        name="Disambiguation + department",
        tier="tier_10_ultrahard",
        anchor="Dept. of Chemistry, U. Washington, Seattle",
        positive="University of Washington",
        negatives=["Washington University in St. Louis", "George Washington University", "Washington State University"],
        notes="Disambiguation with department",
    ),
    RankingTestCase(
        name="NHS + university",
        tier="tier_10_ultrahard",
        anchor="Guy's Hospital, King's Coll. London",
        positive="King's College London",
        negatives=["UCL", "Imperial College", "LSE"],
        notes="Hospital + abbreviated university",
    ),
    RankingTestCase(
        name="VA + medical school",
        tier="tier_10_ultrahard",
        anchor="VA Med Ctr., Harvard Med School",
        positive="Harvard University",
        negatives=["Boston University", "Tufts", "MIT"],
        notes="VA + medical school abbrev",
    ),
    RankingTestCase(
        name="Everything wrong",
        tier="tier_10_ultrahard",
        anchor="dept. phys., harv ard univ., cambridge ma",
        positive="Harvard University",
        negatives=["MIT", "Boston University", "Tufts"],
        notes="Lowercase + OCR + abbrev + location",
    ),
]


ALL_CASES = (
    TIER_1_BASELINE +
    TIER_2_OCR_NOISE +
    TIER_3_ABBREVIATIONS +
    TIER_4_HIERARCHICAL +
    TIER_5_MEDICAL +
    TIER_6_RESEARCH_LABS +
    TIER_7_INTERNATIONAL +
    TIER_8_DISAMBIGUATION +
    TIER_9_NEGATIVE +
    TIER_10_ULTRAHARD
)

TIER_NAMES = {
    "tier_1_baseline": "Baseline",
    "tier_2_ocr_noise": "OCR/Noise",
    "tier_3_abbreviations": "Abbreviations",
    "tier_4_hierarchical": "Hierarchical",
    "tier_5_medical": "Medical/Hospital",
    "tier_6_research_labs": "Research Labs",
    "tier_7_international": "International",
    "tier_8_disambiguation": "Disambiguation",
    "tier_9_negative": "Negative Controls",
    "tier_10_ultrahard": "Ultra-Hard",
}


@dataclass
class ModelResults:
    name: str
    case_results: list[dict]
    accuracy: float
    mrr: float
    mean_positive_score: float
    mean_negative_score: float
    mean_score_gap: float
    tier_accuracy: dict[str, float]
    tier_mrr: dict[str, float]


def evaluate_model(model: CrossEncoder, model_name: str, cases: list[RankingTestCase]) -> ModelResults:
    case_results = []

    for case in cases:
        if case.positive == "NOT_A_MATCH":
            pairs = [[case.anchor, neg] for neg in case.negatives]
            scores = model.predict(pairs)
            max_score = max(scores)
            correct = max_score < 0.5

            case_results.append({
                "name": case.name,
                "tier": case.tier,
                "anchor": case.anchor,
                "positive": case.positive,
                "positive_score": 0.0,
                "best_negative_score": float(max_score),
                "score_gap": -float(max_score),
                "positive_rank": 0,
                "correct": correct,
                "reciprocal_rank": 1.0 if correct else 0.0,
                "notes": case.notes,
            })
            continue

        all_candidates = [case.positive] + case.negatives
        pairs = [[case.anchor, c] for c in all_candidates]

        scores = model.predict(pairs)
        positive_score = scores[0]
        negative_scores = scores[1:]

        ranked = sorted(zip(all_candidates, scores), key=lambda x: -x[1])
        positive_rank = [i for i, (c, s) in enumerate(ranked) if c == case.positive][0] + 1

        best_negative_score = max(negative_scores)
        score_gap = positive_score - best_negative_score

        case_results.append({
            "name": case.name,
            "tier": case.tier,
            "anchor": case.anchor,
            "positive": case.positive,
            "positive_score": float(positive_score),
            "best_negative_score": float(best_negative_score),
            "score_gap": float(score_gap),
            "positive_rank": positive_rank,
            "correct": positive_rank == 1,
            "reciprocal_rank": 1.0 / positive_rank,
            "notes": case.notes,
        })

    accuracy = sum(1 for r in case_results if r["correct"]) / len(case_results)
    mrr = statistics.mean(r["reciprocal_rank"] for r in case_results)

    normal_results = [r for r in case_results if r["positive"] != "NOT_A_MATCH"]
    mean_positive_score = statistics.mean(r["positive_score"] for r in normal_results) if normal_results else 0
    mean_negative_score = statistics.mean(r["best_negative_score"] for r in normal_results) if normal_results else 0
    mean_score_gap = statistics.mean(r["score_gap"] for r in normal_results) if normal_results else 0

    tier_accuracy = {}
    tier_mrr = {}

    for tier in TIER_NAMES.keys():
        tier_results = [r for r in case_results if r["tier"] == tier]
        if tier_results:
            tier_accuracy[tier] = sum(1 for r in tier_results if r["correct"]) / len(tier_results)
            tier_mrr[tier] = statistics.mean(r["reciprocal_rank"] for r in tier_results)

    return ModelResults(
        name=model_name,
        case_results=case_results,
        accuracy=accuracy,
        mrr=mrr,
        mean_positive_score=mean_positive_score,
        mean_negative_score=mean_negative_score,
        mean_score_gap=mean_score_gap,
        tier_accuracy=tier_accuracy,
        tier_mrr=tier_mrr,
    )


def print_comparison(base: ModelResults, finetuned: ModelResults):
    print("=" * 100)
    print("COMPREHENSIVE AFFILIATION RERANKER EVALUATION")
    print("=" * 100)
    print(f"\nTotal test cases: {len(ALL_CASES)}")
    print("\nCases by tier:")
    for tier, name in TIER_NAMES.items():
        count = len([c for c in ALL_CASES if c.tier == tier])
        print(f"  {name}: {count}")

    print("\n" + "=" * 100)
    print("OVERALL METRICS")
    print("=" * 100)
    print(f"\n{'Metric':<30} {'Base Model':<20} {'Fine-tuned':<20} {'Δ':<15}")
    print("-" * 85)

    metrics = [
        ("Accuracy", base.accuracy, finetuned.accuracy, True),
        ("Mean Reciprocal Rank (MRR)", base.mrr, finetuned.mrr, True),
        ("Mean Positive Score", base.mean_positive_score, finetuned.mean_positive_score, True),
        ("Mean Best Negative Score", base.mean_negative_score, finetuned.mean_negative_score, False),
        ("Mean Score Gap", base.mean_score_gap, finetuned.mean_score_gap, True),
    ]

    for name, base_val, ft_val, higher_is_better in metrics:
        delta = ft_val - base_val
        if higher_is_better:
            delta_str = f"+{delta:.4f}" if delta > 0 else f"{delta:.4f}"
            winner = "✓" if delta > 0 else ("" if delta == 0 else "✗")
        else:
            delta_str = f"{delta:.4f}"
            winner = "✓" if delta < 0 else ("" if delta == 0 else "✗")
        print(f"{name:<30} {base_val:<20.4f} {ft_val:<20.4f} {delta_str} {winner}")

    print("\n" + "=" * 100)
    print("ACCURACY BY TIER")
    print("=" * 100)
    print(f"\n{'Tier':<25} {'Base Model':<20} {'Fine-tuned':<20} {'Δ':<15}")
    print("-" * 80)

    for tier, name in TIER_NAMES.items():
        base_acc = base.tier_accuracy.get(tier, 0)
        ft_acc = finetuned.tier_accuracy.get(tier, 0)
        delta = ft_acc - base_acc
        delta_str = f"+{delta:.2%}" if delta > 0 else f"{delta:.2%}"
        winner = "✓" if delta > 0 else ("" if delta == 0 else "✗")
        print(f"{name:<25} {base_acc:<20.2%} {ft_acc:<20.2%} {delta_str} {winner}")

    print("\n" + "=" * 100)
    print("MRR BY TIER")
    print("=" * 100)
    print(f"\n{'Tier':<25} {'Base Model':<20} {'Fine-tuned':<20} {'Δ':<15}")
    print("-" * 80)

    for tier, name in TIER_NAMES.items():
        base_mrr = base.tier_mrr.get(tier, 0)
        ft_mrr = finetuned.tier_mrr.get(tier, 0)
        delta = ft_mrr - base_mrr
        delta_str = f"+{delta:.4f}" if delta > 0 else f"{delta:.4f}"
        winner = "✓" if delta > 0 else ("" if delta == 0 else "✗")
        print(f"{name:<25} {base_mrr:<20.4f} {ft_mrr:<20.4f} {delta_str} {winner}")

    print("\n" + "=" * 100)
    print("FAILURE ANALYSIS")
    print("=" * 100)

    base_failures = [r for r in base.case_results if not r["correct"]]
    ft_failures = [r for r in finetuned.case_results if not r["correct"]]

    print(f"\nBase model failures: {len(base_failures)}/{len(base.case_results)}")
    for tier in TIER_NAMES.keys():
        tier_failures = [f for f in base_failures if f["tier"] == tier]
        if tier_failures:
            print(f"\n  {TIER_NAMES[tier]}:")
            for f in tier_failures[:5]:  # Limit to 5 per tier
                print(f"    - {f['name']}: ranked #{f['positive_rank']}")
            if len(tier_failures) > 5:
                print(f"    ... and {len(tier_failures) - 5} more")

    print(f"\nFine-tuned failures: {len(ft_failures)}/{len(finetuned.case_results)}")
    for tier in TIER_NAMES.keys():
        tier_failures = [f for f in ft_failures if f["tier"] == tier]
        if tier_failures:
            print(f"\n  {TIER_NAMES[tier]}:")
            for f in tier_failures[:5]:
                print(f"    - {f['name']}: ranked #{f['positive_rank']}")
            if len(tier_failures) > 5:
                print(f"    ... and {len(tier_failures) - 5} more")

    print("\n" + "=" * 100)
    print("IMPROVEMENT ANALYSIS")
    print("=" * 100)

    improved = []
    regressed = []

    for b, f in zip(base.case_results, finetuned.case_results):
        if f["correct"] and not b["correct"]:
            improved.append((b, f))
        elif b["correct"] and not f["correct"]:
            regressed.append((b, f))

    print(f"\nCases where fine-tuned IMPROVED over base: {len(improved)}")
    for tier in TIER_NAMES.keys():
        tier_improved = [(b, f) for b, f in improved if b["tier"] == tier]
        if tier_improved:
            print(f"\n  {TIER_NAMES[tier]}:")
            for b, f in tier_improved[:5]:
                print(f"    ✓ {b['name']}")
            if len(tier_improved) > 5:
                print(f"    ... and {len(tier_improved) - 5} more")

    print(f"\nCases where fine-tuned REGRESSED from base: {len(regressed)}")
    for tier in TIER_NAMES.keys():
        tier_regressed = [(b, f) for b, f in regressed if b["tier"] == tier]
        if tier_regressed:
            print(f"\n  {TIER_NAMES[tier]}:")
            for b, f in tier_regressed:
                print(f"    ✗ {b['name']}")

    # Summary
    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    print(f"""
Base Model:     {base.name}
Fine-tuned:     {finetuned.name}

Overall Accuracy:   Base {base.accuracy:.1%} vs Fine-tuned {finetuned.accuracy:.1%} ({'+' if finetuned.accuracy >= base.accuracy else ''}{(finetuned.accuracy - base.accuracy):.1%})
Overall MRR:        Base {base.mrr:.4f} vs Fine-tuned {finetuned.mrr:.4f} ({'+' if finetuned.mrr >= base.mrr else ''}{(finetuned.mrr - base.mrr):.4f})

Score Calibration:
  - Positive scores: Base avg {base.mean_positive_score:.3f} vs Fine-tuned avg {finetuned.mean_positive_score:.3f}
  - Negative scores: Base avg {base.mean_negative_score:.3f} vs Fine-tuned avg {finetuned.mean_negative_score:.3f}
  - Score gap:       Base avg {base.mean_score_gap:.3f} vs Fine-tuned avg {finetuned.mean_score_gap:.3f}

Net Change: {len(improved)} improvements, {len(regressed)} regressions
""")

    return {
        "improved": improved,
        "regressed": regressed,
    }


def main():
    print("Loading models...")

    base_model = CrossEncoder(
        "jinaai/jina-reranker-v2-base-multilingual",
        trust_remote_code=True,
    )

    finetuned_model = CrossEncoder(
        "cometadata/jina-reranker-v2-multilingual-affiliations",
        trust_remote_code=True,
    )

    print(f"\nEvaluating {len(ALL_CASES)} test cases across {len(TIER_NAMES)} tiers...")

    print("\nEvaluating base model...")
    base_results = evaluate_model(
        base_model,
        "jinaai/jina-reranker-v2-base-multilingual",
        ALL_CASES,
    )

    print("Evaluating fine-tuned model...")
    finetuned_results = evaluate_model(
        finetuned_model,
        "cometadata/jina-reranker-v2-multilingual-affiliations",
        ALL_CASES,
    )

    comparison = print_comparison(base_results, finetuned_results)

    output = {
        "base_model": base_results.name,
        "finetuned_model": finetuned_results.name,
        "num_cases": len(ALL_CASES),
        "tiers": {tier: {"name": name, "count": len([c for c in ALL_CASES if c.tier == tier])}
                  for tier, name in TIER_NAMES.items()},
        "base_results": {
            "accuracy": base_results.accuracy,
            "mrr": base_results.mrr,
            "mean_positive_score": base_results.mean_positive_score,
            "mean_negative_score": base_results.mean_negative_score,
            "mean_score_gap": base_results.mean_score_gap,
            "tier_accuracy": base_results.tier_accuracy,
            "tier_mrr": base_results.tier_mrr,
            "case_results": base_results.case_results,
        },
        "finetuned_results": {
            "accuracy": finetuned_results.accuracy,
            "mrr": finetuned_results.mrr,
            "mean_positive_score": finetuned_results.mean_positive_score,
            "mean_negative_score": finetuned_results.mean_negative_score,
            "mean_score_gap": finetuned_results.mean_score_gap,
            "tier_accuracy": finetuned_results.tier_accuracy,
            "tier_mrr": finetuned_results.tier_mrr,
            "case_results": finetuned_results.case_results,
        },
    }

    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            import numpy as np
            if isinstance(obj, (np.bool_, np.integer)):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)

    with open("eval_results.json", "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)

    print("\nDetailed results saved to eval_results.json")


if __name__ == "__main__":
    main()
