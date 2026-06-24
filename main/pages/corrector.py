import language_tool_python

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False


# ---------------------------------------------------------------------------
# Lookup tables used by rule helpers
# ---------------------------------------------------------------------------

# Masculine nouns that misleadingly start with a vowel or take "la/le"
# (add more as needed)
_MASCULINE_NOUNS = {
    "problema", "programma", "sistema", "tema", "clima", "schema",
    "panorama", "diploma", "dramma", "poema", "aroma",
}

# Feminine nouns that misleadingly look masculine
_FEMININE_NOUNS = {
    "mano", "radio", "foto", "moto", "auto",
}

# Verbs that form the passato prossimo with "essere" (intransitive/reflexive)
_ESSERE_VERBS = {
    "andare", "venire", "partire", "arrivare", "tornare", "uscire",
    "entrare", "nascere", "morire", "essere", "stare", "restare",
    "rimanere", "cadere", "salire", "scendere", "diventare", "sembrare",
    "parere", "succedere", "accadere", "capitare", "piacere", "dispiacere",
    "mancare", "bastare", "costare", "durare", "dipendere", "apparire",
}

# Preposition contractions: prep + article → contracted form
# Used to detect *wrong* contractions (user wrote articulated when they
# should not, or wrote the plain form when they should have contracted).
_PREP_ARTICLE_CONTRACTIONS = {
    ("di", "il"):  "del",
    ("di", "lo"):  "dello",
    ("di", "la"):  "della",
    ("di", "i"):   "dei",
    ("di", "gli"): "degli",
    ("di", "le"):  "delle",
    ("di", "l'"):  "dell'",
    ("a",  "il"):  "al",
    ("a",  "lo"):  "allo",
    ("a",  "la"):  "alla",
    ("a",  "i"):   "ai",
    ("a",  "gli"): "agli",
    ("a",  "le"):  "alle",
    ("a",  "l'"):  "all'",
    ("da", "il"):  "dal",
    ("da", "lo"):  "dallo",
    ("da", "la"):  "dalla",
    ("da", "i"):   "dai",
    ("da", "gli"): "dagli",
    ("da", "le"):  "dalle",
    ("da", "l'"):  "dall'",
    ("in", "il"):  "nel",
    ("in", "lo"):  "nello",
    ("in", "la"):  "nella",
    ("in", "i"):   "nei",
    ("in", "gli"): "negli",
    ("in", "le"):  "nelle",
    ("in", "l'"):  "nell'",
    ("su", "il"):  "sul",
    ("su", "lo"):  "sullo",
    ("su", "la"):  "sulla",
    ("su", "i"):   "sui",
    ("su", "gli"): "sugli",
    ("su", "le"):  "sulle",
    ("su", "l'"):  "sull'",
}

# Reverse map: contracted form → (prep, article)
_CONTRACTED_TO_PARTS = {v: k for k, v in _PREP_ARTICLE_CONTRACTIONS.items()}

# Prepositions that should NOT be contracted with a following article
# (e.g. "con" is usually left separate in modern Italian)
_NON_CONTRACTING_PREPS = {"con", "per", "tra", "fra", "su"}  # "su" does contract

# Verbs whose argument must be introduced by a specific preposition
# Format: lemma → {"expected": prep, "wrong": [list of wrong preps], "example": ...}
_VERB_PREP_RULES = {
    "andare": {
        "expected": "a",
        "wrong": ["in", "al"],   # "vado in casa" wrong; "vado in" before cities is fine
        "note": "Motion to a place: 'andare a + inf' or 'andare a + city'.",
    },
    "pensare": {
        "expected": "a",
        "wrong": ["di"],
        "note": "Pensare A qualcosa (think about); pensare DI fare (intend to do).",
    },
    "ringraziare": {
        "expected": "per",
        "wrong": ["di"],
        "note": "Ringraziare PER qualcosa.",
    },
    "dipendere": {
        "expected": "da",
        "wrong": ["di", "a"],
        "note": "Dipendere DA qualcosa/qualcuno.",
    },
    "parlare": {
        "expected": "di",
        "wrong": ["su", "a"],
        "note": "Parlare DI qualcosa.",
    },
}


class GrammarCorrector:
    """
    A unified grammar correction class that uses LanguageTool for general
    spell-checking, punctuation, and structural issues, and utilises
    spaCy NLP dependency parsing to flag granular Italian agreement errors.

    Rules implemented
    -----------------
    LanguageTool (external):
        • Spelling, punctuation, typography, casing, general grammar.

    spaCy (dependency-parse based):
        1. Noun agreement – determiner / adjective ↔ noun
           (e.g. ``qualche libri``, ``bella ragazzo``)
        2. Subject–verb agreement
           (e.g. ``Gli studenti va a scuola``)
        3. Possessive–noun agreement          [NEW]
           (e.g. ``nostri progetto``)
        4. Auxiliary–past-participle agreement [NEW]
           (e.g. ``siamo tornato``)
        5. Article–noun agreement              [NEW]
           (e.g. ``un ragazza``, ``la problema``)
        6. Post-nominal adjective agreement    [NEW]
           (e.g. ``case grande``, ``problemi complesso``)
        7. Preposition contraction errors      [NEW]
           (e.g. ``di il libro`` → should be ``del libro``)
        8. Verb–preposition collocation errors [NEW]
           (e.g. ``vado in casa`` instead of ``vado a casa``)
    """

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def __init__(self, use_spacy: bool = True):
        """
        Initialise LanguageTool for Italian ("it") and optionally load the
        large Italian spaCy model.
        """
        self.tool = language_tool_python.LanguageTool("it")

        self.nlp = None
        if use_spacy and SPACY_AVAILABLE:
            try:
                self.nlp = spacy.load("it_core_news_lg")
            except OSError:
                print(
                    "spaCy model not found. Run:\n"
                    "python -m spacy download it_core_news_lg"
                )

    # ------------------------------------------------------------------
    # Generic helpers
    # ------------------------------------------------------------------

    def get_match_value(self, match, *names, default=None):
        """Safely extract an attribute from a LanguageTool match object."""
        for name in names:
            if hasattr(match, name):
                return getattr(match, name)
        return default

    def classify_lt_issue(self, category: str, rule: str) -> str:
        """Map LanguageTool category/rule identifiers to canonical issue types."""
        category = (category or "").upper()
        rule = (rule or "").upper()

        if category in ("PUNCTUATION", "TYPOGRAPHY", "CASING"):
            return "punctuation"
        if any(k in rule for k in ("COMMA", "APOSTROPHE", "WHITESPACE", "PUNCT")):
            return "punctuation"
        if category in ("TYPOS", "MISSPELLING"):
            return "spelling"
        return "grammar"

    def _issue(
        self,
        *,
        text: str,
        token,
        rule: str,
        message: str,
        suggestions: list = None,
    ) -> dict:
        """
        Build a standardised issue dictionary from a spaCy token and metadata.
        Centralises the repeated boilerplate across all spaCy rules.
        """
        return {
            "source": "spaCy",
            "issue_type": "grammar",
            "message": message,
            "rule": rule,
            "category": "GRAMMAR",
            "offset": token.idx,
            "length": len(token.text),
            "wrong_text": token.text,
            "context": text[max(0, token.idx - 30): token.idx + 40],
            "suggestions": suggestions or [],
        }

    # ------------------------------------------------------------------
    # LanguageTool result parsing
    # ------------------------------------------------------------------

    def parse_language_tool_matches(self, text: str, matches: list) -> list:
        """
        Transform raw LanguageTool match objects into a standardised list of
        dictionaries, capping replacement suggestions at five per match.
        """
        parsed = []
        for match in matches:
            offset = self.get_match_value(match, "offset", default=0)
            length = self.get_match_value(
                match, "errorLength", "error_length", default=0
            )
            category = self.get_match_value(match, "category", default="")
            rule = self.get_match_value(match, "ruleId", "rule_id", default="")

            parsed.append(
                {
                    "source": "LanguageTool",
                    "issue_type": self.classify_lt_issue(category, rule),
                    "message": self.get_match_value(match, "message", default=""),
                    "rule": rule,
                    "category": category,
                    "offset": offset,
                    "length": length,
                    "wrong_text": text[offset: offset + length],
                    "context": self.get_match_value(match, "context", default=""),
                    "suggestions": self.get_match_value(
                        match, "replacements", default=[]
                    )[:5],
                }
            )
        return parsed

    # ------------------------------------------------------------------
    # spaCy rule 1 – determiner / adjective ↔ noun agreement
    # ------------------------------------------------------------------

    def spacy_noun_agreement_issues(self, text: str) -> list:
        """
        Detect gender/number mismatches between a pre-nominal determiner or
        adjective and its governing noun.

        Examples:  ``qualche libri``,  ``bella ragazzo``
        """
        if self.nlp is None:
            return []

        issues = []
        doc = self.nlp(text)

        for token in doc:
            head = token.head
            if token.dep_ in ("det", "amod") and head.pos_ in ("NOUN", "PROPN"):
                t_gender = token.morph.get("Gender")
                t_number = token.morph.get("Number")
                h_gender = head.morph.get("Gender")
                h_number = head.morph.get("Number")

                mismatches = []
                if t_gender and h_gender and t_gender != h_gender:
                    mismatches.append("gender agreement")
                if t_number and h_number and t_number != h_number:
                    mismatches.append("number agreement")

                if mismatches:
                    issues.append(
                        self._issue(
                            text=text,
                            token=token,
                            rule="SPACY_NOUN_AGREEMENT",
                            message=(
                                f"Possible {' and '.join(mismatches)} issue: "
                                f"'{token.text}' may not agree with '{head.text}'."
                            ),
                        )
                    )
        return issues

    # ------------------------------------------------------------------
    # spaCy rule 2 – subject–verb agreement
    # ------------------------------------------------------------------

    def spacy_subject_verb_issues(self, text: str) -> list:
        """
        Detect number mismatches between a nominal subject and its verb.

        Example:  ``Gli studenti va a scuola``  (studenti=Plur, va=Sing)
        """
        if self.nlp is None:
            return []

        issues = []
        doc = self.nlp(text)

        for token in doc:
            if token.dep_ in ("nsubj", "nsubj:pass") and token.head.pos_ in (
                "VERB",
                "AUX",
            ):
                subj = token
                verb = token.head
                s_number = subj.morph.get("Number")
                v_number = verb.morph.get("Number")

                if s_number and v_number and s_number != v_number:
                    issues.append(
                        self._issue(
                            text=text,
                            token=verb,
                            rule="SPACY_SUBJECT_VERB_AGREEMENT",
                            message=(
                                f"Possible subject-verb agreement error: "
                                f"'{subj.text}' is {s_number[0]}, "
                                f"but '{verb.text}' is {v_number[0]}."
                            ),
                        )
                    )
        return issues

    # ------------------------------------------------------------------
    # spaCy rule 3 – possessive ↔ noun agreement  [NEW]
    # ------------------------------------------------------------------

    def spacy_possessive_noun_issues(self, text: str) -> list:
        """
        Detect gender/number mismatches between a possessive pronoun/determiner
        and its head noun.

        Example:  ``nostri progetto``  (nostri=Masc,Plur  progetto=Masc,Sing)

        spaCy labels possessives with dep_='det:poss' or pos_='DET' and the
        morph feature Poss=Yes.
        """
        if self.nlp is None:
            return []

        issues = []
        doc = self.nlp(text)

        for token in doc:
            is_possessive = (
                token.dep_ == "det:poss"
                or (token.pos_ == "DET" and "Yes" in token.morph.get("Poss", []))
            )
            if not is_possessive:
                continue

            head = token.head
            if head.pos_ not in ("NOUN", "PROPN"):
                continue

            t_gender = token.morph.get("Gender")
            t_number = token.morph.get("Number")
            h_gender = head.morph.get("Gender")
            h_number = head.morph.get("Number")

            mismatches = []
            if t_gender and h_gender and t_gender != h_gender:
                mismatches.append("gender")
            if t_number and h_number and t_number != h_number:
                mismatches.append("number")

            if mismatches:
                issues.append(
                    self._issue(
                        text=text,
                        token=token,
                        rule="SPACY_POSSESSIVE_NOUN_AGREEMENT",
                        message=(
                            f"Possessive–noun {' and '.join(mismatches)} mismatch: "
                            f"'{token.text}' does not agree with '{head.text}'. "
                            f"(e.g. 'nostri progetto' → 'nostro progetto')"
                        ),
                    )
                )
        return issues

    # ------------------------------------------------------------------
    # spaCy rule 4 – auxiliary ↔ past-participle agreement  [NEW]
    # ------------------------------------------------------------------

    def spacy_aux_participle_issues(self, text: str) -> list:
        """
        Detect gender/number mismatches between an *essere*-auxiliary and its
        past participle in compound tenses.

        Italian rule: when the auxiliary is *essere*, the past participle must
        agree in gender and number with the subject.

        Example:  ``siamo tornato``  (noi=Masc,Plur  tornato=Masc,Sing)

        Detection strategy
        ------------------
        1. Find VERB tokens whose VerbForm=Part (past participle).
        2. Check whether any AUX sibling (same head) is a form of *essere*.
        3. Find the clause's nominal subject.
        4. Compare subject morph with participle morph.
        """
        if self.nlp is None:
            return []

        issues = []
        doc = self.nlp(text)

        for token in doc:
            # Target: past participles
            if not (
                token.pos_ == "VERB"
                and "Part" in token.morph.get("VerbForm", [])
            ):
                continue

            # Find an essere-auxiliary in the same clause
            aux_essere = None
            for child in token.children:
                if child.dep_ == "aux" and child.lemma_.lower() in (
                    "essere", "venire"
                ):
                    aux_essere = child
                    break
            # Also check if the participle's head is an essere auxiliary
            if aux_essere is None and token.head.lemma_.lower() in ("essere", "venire"):
                aux_essere = token.head

            if aux_essere is None:
                continue

            # Find the subject of this predicate
            subject = None
            for child in token.children:
                if child.dep_ in ("nsubj", "nsubj:pass"):
                    subject = child
                    break
            # Subject might hang off the auxiliary instead
            if subject is None:
                for child in aux_essere.children:
                    if child.dep_ in ("nsubj", "nsubj:pass"):
                        subject = child
                        break

            if subject is None:
                continue

            s_gender = subject.morph.get("Gender")
            s_number = subject.morph.get("Number")
            p_gender = token.morph.get("Gender")
            p_number = token.morph.get("Number")

            mismatches = []
            if s_gender and p_gender and s_gender != p_gender:
                mismatches.append("gender")
            if s_number and p_number and s_number != p_number:
                mismatches.append("number")

            if mismatches:
                issues.append(
                    self._issue(
                        text=text,
                        token=token,
                        rule="SPACY_AUX_PARTICIPLE_AGREEMENT",
                        message=(
                            f"Auxiliary–participle {' and '.join(mismatches)} mismatch: "
                            f"subject '{subject.text}' is "
                            f"{', '.join(s_gender + s_number)}, but participle "
                            f"'{token.text}' is {', '.join(p_gender + p_number)}. "
                            f"(e.g. 'siamo tornato' → 'siamo tornati')"
                        ),
                    )
                )
        return issues

    # ------------------------------------------------------------------
    # spaCy rule 5 – article ↔ noun agreement  [NEW]
    # ------------------------------------------------------------------

    def spacy_article_noun_issues(self, text: str) -> list:
        """
        Detect mismatches between a definite/indefinite article and its noun.

        Covers:
        • Gender mismatch  →  ``un ragazza`` (un=Masc, ragazza=Fem)
        • Exception nouns  →  ``la problema`` (problema is Masc despite -a ending)

        Strategy: look for DET tokens with PronType=Art and compare morph with
        their head noun, cross-referencing the exception dictionaries above.
        """
        if self.nlp is None:
            return []

        issues = []
        doc = self.nlp(text)

        for token in doc:
            if not (
                token.pos_ == "DET"
                and "Art" in token.morph.get("PronType", [])
            ):
                continue

            head = token.head
            if head.pos_ not in ("NOUN", "PROPN"):
                continue

            t_gender = token.morph.get("Gender")
            t_number = token.morph.get("Number")
            h_gender = head.morph.get("Gender")
            h_number = head.morph.get("Number")

            mismatches = []

            # Check exception nouns where the model might parse gender wrongly
            lemma = head.lemma_.lower()
            if lemma in _MASCULINE_NOUNS and t_gender == ["Fem"]:
                mismatches.append("gender (noun is masculine despite -a ending)")
            elif lemma in _FEMININE_NOUNS and t_gender == ["Masc"]:
                mismatches.append("gender (noun is feminine despite appearance)")
            else:
                if t_gender and h_gender and t_gender != h_gender:
                    mismatches.append("gender")
                if t_number and h_number and t_number != h_number:
                    mismatches.append("number")

            if mismatches:
                issues.append(
                    self._issue(
                        text=text,
                        token=token,
                        rule="SPACY_ARTICLE_NOUN_AGREEMENT",
                        message=(
                            f"Article–noun {' and '.join(mismatches)} mismatch: "
                            f"'{token.text}' does not agree with '{head.text}'. "
                            f"(e.g. 'un ragazza' → 'una ragazza', "
                            f"'la problema' → 'il problema')"
                        ),
                    )
                )
        return issues

    # ------------------------------------------------------------------
    # spaCy rule 6 – post-nominal adjective agreement  [NEW]
    # ------------------------------------------------------------------

    def spacy_postnominal_adjective_issues(self, text: str) -> list:
        """
        Detect gender/number mismatches for adjectives that follow their noun
        (post-nominal position).

        Examples:
            ``case grande``   →  ``case grandi``
            ``problemi complesso``  →  ``problemi complessi``

        spaCy marks these as amod with the adjective appearing *after* the noun
        in the token sequence (token.i > head.i).
        """
        if self.nlp is None:
            return []

        issues = []
        doc = self.nlp(text)

        for token in doc:
            if not (
                token.dep_ == "amod"
                and token.head.pos_ in ("NOUN", "PROPN")
                and token.i > token.head.i          # adjective comes after noun
            ):
                continue

            head = token.head
            t_gender = token.morph.get("Gender")
            t_number = token.morph.get("Number")
            h_gender = head.morph.get("Gender")
            h_number = head.morph.get("Number")

            mismatches = []
            if t_gender and h_gender and t_gender != h_gender:
                mismatches.append("gender")
            if t_number and h_number and t_number != h_number:
                mismatches.append("number")

            if mismatches:
                issues.append(
                    self._issue(
                        text=text,
                        token=token,
                        rule="SPACY_POSTNOMINAL_ADJ_AGREEMENT",
                        message=(
                            f"Post-nominal adjective {' and '.join(mismatches)} "
                            f"mismatch: '{token.text}' does not agree with "
                            f"'{head.text}'. "
                            f"(e.g. 'case grande' → 'case grandi')"
                        ),
                    )
                )
        return issues

    # ------------------------------------------------------------------
    # spaCy rule 7 – preposition contraction errors  [NEW]
    # ------------------------------------------------------------------

    def spacy_preposition_contraction_issues(self, text: str) -> list:
        """
        Detect places where a preposition + article should be written as a
        contracted articulated preposition (preposizione articolata) but is not.

        Example:  ``di il libro``  →  should be  ``del libro``

        Detection: walk the token list looking for a preposition immediately
        followed by an article where the contracted form is known.
        """
        if self.nlp is None:
            return []

        issues = []
        doc = self.nlp(text)
        tokens = list(doc)

        for i, token in enumerate(tokens[:-1]):
            next_tok = tokens[i + 1]

            prep = token.text.lower()
            art = next_tok.text.lower()

            contracted = _PREP_ARTICLE_CONTRACTIONS.get((prep, art))
            if contracted is None:
                continue

            # If they are syntactically related (prep → det or prep → head of det)
            # we flag the pair.
            issues.append(
                self._issue(
                    text=text,
                    token=token,
                    rule="SPACY_PREP_CONTRACTION",
                    message=(
                        f"Preposition + article should be contracted: "
                        f"'{token.text} {next_tok.text}' → '{contracted}'."
                    ),
                    suggestions=[contracted],
                )
            )
        return issues

    # ------------------------------------------------------------------
    # spaCy rule 8 – verb–preposition collocation errors  [NEW]
    # ------------------------------------------------------------------

    def spacy_verb_preposition_issues(self, text: str) -> list:
        """
        Detect verbs used with a wrong preposition based on known collocation
        rules.

        Examples:
            ``vado in casa``  →  ``vado a casa``
            ``pensare su qualcosa``  →  ``pensare a qualcosa``

        Strategy: find verbs whose lemma is in ``_VERB_PREP_RULES``, then look
        at their prepositional children (dep_='obl' or 'nmod') and check the
        governing preposition token.
        """
        if self.nlp is None:
            return []

        issues = []
        doc = self.nlp(text)

        for token in doc:
            if token.pos_ not in ("VERB", "AUX"):
                continue

            rule_entry = _VERB_PREP_RULES.get(token.lemma_.lower())
            if rule_entry is None:
                continue

            expected_prep = rule_entry["expected"]
            wrong_preps = rule_entry["wrong"]
            note = rule_entry.get("note", "")

            for child in token.children:
                if child.dep_ not in ("obl", "obl:agent", "nmod", "advmod"):
                    continue
                # The preposition is the 'case' child of the oblique
                for grandchild in child.children:
                    if grandchild.dep_ == "case" and grandchild.pos_ == "ADP":
                        used_prep = grandchild.text.lower()
                        if used_prep in wrong_preps:
                            issues.append(
                                self._issue(
                                    text=text,
                                    token=grandchild,
                                    rule="SPACY_VERB_PREP_COLLOCATION",
                                    message=(
                                        f"Wrong preposition after '{token.text}': "
                                        f"used '{used_prep}', expected "
                                        f"'{expected_prep}'. {note}"
                                    ),
                                    suggestions=[expected_prep],
                                )
                            )
        return issues

    # ------------------------------------------------------------------
    # Main orchestration
    # ------------------------------------------------------------------

    def correct_text(self, text: str) -> dict:
        """
        Main entry point.

        Runs LanguageTool and all eight spaCy rules, merges the results, and
        returns a unified audit payload.

        Returns
        -------
        dict with keys:
            original  – the input string unchanged
            corrected – the LanguageTool auto-corrected version
            polished  – same as corrected (hook for future post-processing)
            matches   – list of standardised issue dictionaries
        """
        if not text or not text.strip():
            return {
                "original": text,
                "corrected": text,
                "polished": text,
                "matches": [],
            }

        # --- LanguageTool ---
        lt_matches = self.tool.check(text)
        corrected_text = language_tool_python.utils.correct(text, lt_matches)
        parsed_lt = self.parse_language_tool_matches(text, lt_matches)

        # --- spaCy rules ---
        spacy_issues = (
            self.spacy_noun_agreement_issues(text)           # rule 1
            + self.spacy_subject_verb_issues(text)           # rule 2
            + self.spacy_possessive_noun_issues(text)        # rule 3
            + self.spacy_aux_participle_issues(text)         # rule 4
            + self.spacy_article_noun_issues(text)           # rule 5
            + self.spacy_postnominal_adjective_issues(text)  # rule 6
            + self.spacy_preposition_contraction_issues(text)# rule 7
            + self.spacy_verb_preposition_issues(text)       # rule 8
        )

        return {
            "original": text,
            "corrected": corrected_text,
            "polished": corrected_text,
            "matches": parsed_lt + spacy_issues,
        }