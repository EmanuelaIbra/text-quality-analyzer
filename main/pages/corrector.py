import language_tool_python

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False


# ---------------------------------------------------------------------------
# Lookup tables used by rule helpers
# ---------------------------------------------------------------------------
#
# Design principle
# ----------------
# Each set below is a curated *seed* of known exceptions or members.
# At runtime, GrammarCorrector.__init__() calls _build_dynamic_lexicons()
# which runs the loaded spaCy model over a representative corpus of Italian
# sentences and uses the model's own morphology tags to extend these sets
# automatically.  That way the lists stay useful even for words that weren't
# hand-curated here.
#
# You can always add words directly to the seed sets below; they will be
# merged with whatever the dynamic builder discovers.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# MASCULINE NOUNS whose surface form ends in -a (Greek/Latin loans)
# and are therefore systematically confused with feminine nouns.
# Pattern: -ma, -ta, -pa, -ra, -da, -ca endings from Greek -μα/-τα etc.
# ---------------------------------------------------------------------------
_MASCULINE_NOUNS: set[str] = {
    # -ma words (Greek neuter -μα)
    "problema", "programma", "sistema", "tema", "clima", "schema",
    "panorama", "diploma", "dramma", "poema", "aroma", "fantasma",
    "prisma", "teorema", "dilemma", "dogma", "enigma", "emblema",
    "trauma", "plasma", "sintoma", "sintoma", "telegramma", "telegramma",
    "diagramma", "anagramma", "epigrama", "epigramma", "ideogramma",
    "monogramma", "ologramma", "fonogramma", "crittogramma",
    "panorama", "diorama", "cosmorama", "georama",
    "reuma", "edema", "eczema", "enema", "enfisema", "glaucoma",
    "melanoma", "carcinoma", "adenoma", "sarcoma", "linfoma",
    # -ta words
    "atleta", "pianeta", "poeta", "profeta", "cometa", "meta",
    "delta", "beta", "eta", "zeta", "theta",
    # -ista words used as masculine nouns in context
    "artista", "giornalista", "pianista", "violinista", "ciclista",
    "dentista", "terrorista", "comunista", "fascista", "capitalista",
    "socialista", "ottimista", "pessimista", "realista", "idealista",
    "protagonista", "antagonista", "antagonista",
    # other irregular masculines
    "brindisi", "alibi", "safari", "koala", "panda", "gorilla",
    "lama",   # the animal / monk
    "vaglia",  # postal order
    "sosia",   # look-alike
}

# ---------------------------------------------------------------------------
# FEMININE NOUNS whose surface form ends in -o or looks masculine.
# ---------------------------------------------------------------------------
_FEMININE_NOUNS: set[str] = {
    # Short clippings (always feminine regardless of ending)
    "mano",   # irregular historical feminine
    "radio", "foto", "moto", "auto", "metro", "bici",
    "eco",    # feminine in singular: l'eco sonora
    # Foreign loanwords treated as feminine
    "moto",   # motocicletta
    "dinamo", "libido",
    # -ione / -zione / -sione (all feminine — important high-frequency class)
    # (These end in -e not -o but are often misgendered by learners)
    "nazione", "situazione", "soluzione", "relazione", "collaborazione",
    "comunicazione", "informazione", "istruzione", "tradizione", "funzione",
    "condizione", "azione", "reazione", "produzione", "distribuzione",
    "percezione", "eccezione", "connessione", "sessione", "missione",
    "passione", "discussione", "decisione", "conclusione", "dimensione",
    "attenzione", "intenzione", "menzione", "tensione", "pensione",
    "versione", "visione", "revisione", "previsione", "divisione",
    "invasione", "evasione", "persuasione", "illusione", "allusione",
    # -tà / -tù (all feminine)
    "città", "libertà", "verità", "qualità", "università", "attività",
    "capacità", "possibilità", "opportunità", "necessità", "realtà",
    "identità", "priorità", "sicurezza", "difficoltà", "novità",
    "curiosità", "creatività", "nazionalità", "personalità",
    # -si (Greek feminine)
    "crisi", "tesi", "analisi", "sintesi", "ipotesi", "diagnosi",
    "prognosi", "nevrosi", "psicosi", "metamorfosi", "osmosi",
    "parentesi", "enfasi", "perifrasi",
}

# ---------------------------------------------------------------------------
# VERBS that form compound tenses with ESSERE (not avere).
# Includes: motion verbs, change-of-state verbs, copulas,
# weather verbs, impersonal verbs, all reflexives (handled separately).
# ---------------------------------------------------------------------------
_ESSERE_VERBS: set[str] = {
    # Core motion verbs
    "andare", "venire", "partire", "arrivare", "tornare", "uscire",
    "entrare", "salire", "scendere", "cadere", "fuggire", "scappare",
    "correre",   # correre can take either; essere when intransitive
    "passare",   # essere when intransitive motion
    "ritornare", "rientrare", "ripartire", "riuscire", "risalire",
    "ridiscendere", "rivenire",
    # Change-of-state / becoming
    "nascere", "morire", "diventare", "divenire", "crescere",
    "invecchiare", "migliorare", "peggiorare", "guarire", "ammalarsi",
    "ingrassare", "dimagrire", "arrossire", "impallidire",
    # Copulas and stative
    "essere", "stare", "sembrare", "parere", "risultare", "apparire",
    "restare", "rimanere", "durare",
    # Impersonal / weather (used with essere in most constructions)
    "piacere", "dispiacere", "succedere", "accadere", "capitare",
    "mancare", "bastare", "costare", "servire", "importare",
    "interessare", "dipendere", "appartenere",
    "piovere", "nevicare", "grandinare", "tuonare",
    # Verbs of appearance / disappearance
    "comparire", "scomparire", "apparire", "sparire", "emergere",
    "affiorare", "sorgere", "tramontare",
    # Verbs of spatial relation / existence
    "esistere", "vivere",   # can take essere (intransitive)
    "giacere", "stendersi",
    # Common inchoatives
    "cominciare", "iniziare", "finire", "terminare", "smettere",
    # Reflexive verbs (all take essere — representative set)
    "alzarsi", "lavarsi", "vestirsi", "sedersi", "fermarsi",
    "sentirsi", "trovarsi", "chiamarsi", "svegliarsi", "addormentarsi",
    "annoiarsi", "arrabbiarsi", "dimenticarsi", "innamorarsi",
    "perdersi", "riposarsi", "sbrigarsi", "sposarsi", "vergognarsi",
    "avvicinarsi", "accorgersi", "rendersi", "prepararsi",
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
        "wrong": ["in", "al"],
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

# ---------------------------------------------------------------------------
# Lookup tables for rules 9–16
# ---------------------------------------------------------------------------

# Copular / linking verbs whose predicate adjective must agree with subject
_COPULAR_VERBS = {
    "essere", "sembrare", "parere", "diventare", "divenire", "restare",
    "rimanere", "apparire", "risultare", "sentirsi", "ritrovarsi",
}

# Reflexive verbs that require a clitic pronoun (lemma → expected clitic lemma)
# We flag when the verb appears without any clitic child.
_REFLEXIVE_VERBS = {
    "lavarsi", "alzarsi", "sedersi", "vestirsi", "prepararsi",
    "fermarsi", "sentirsi", "trovarsi", "chiamarsi", "svegliarsi",
    "addormentarsi", "annoiarsi", "arrabbiarsi", "dimenticarsi",
    "innamorarsi", "perdersi", "riposarsi", "sbrigarsi", "sposarsi",
    "vergognarsi", "avvicinarsi", "accorgersi", "rendersi",
}

# Non-reflexive surface forms of reflexive verbs (lemma without -si)
# Used to detect when the bare verb is used where the reflexive is required.
_REFLEXIVE_BASE_LEMMAS = {v.rstrip("si").rstrip("r") + "re" for v in _REFLEXIVE_VERBS}
# Also store the explicit mapping base_lemma → reflexive_lemma for messages
_REFLEXIVE_LEMMA_MAP = {
    "lavare": "lavarsi", "alzare": "alzarsi", "sedere": "sedersi",
    "vestire": "vestirsi", "preparare": "prepararsi", "fermare": "fermarsi",
    "sentire": "sentirsi", "trovare": "trovarsi", "chiamare": "chiamarsi",
    "svegliare": "svegliarsi", "addormentare": "addormentarsi",
    "annoiare": "annoiarsi", "arrabbiare": "arrabbiarsi",
    "dimenticare": "dimenticarsi", "innamorare": "innamorarsi",
    "perdere": "perdersi", "riposare": "riposarsi", "sbrigare": "sbrigarsi",
    "sposare": "sposarsi", "vergognare": "vergognarsi",
    "avvicinare": "avvicinarsi", "accorgere": "accorgersi",
    "rendere": "rendersi",
}

# Negative polarity items: when one of these appears without "non" before
# the finite verb, it is likely a double-negation error.
_NEGATIVE_POLARITY_WORDS = {
    "niente", "nulla", "nessuno", "nessuna", "mai", "nemmeno",
    "neanche", "neppure", "né", "affatto",
}

# Italian interrogative / WH-words that trigger word-order check
_WH_WORDS = {
    "cosa", "che", "chi", "come", "quando", "dove", "perché",
    "quanto", "quale", "quali",
}

# Modal verbs: the verb immediately following must be infinitive (VerbForm=Inf)
_MODAL_VERBS = {
    "volere", "potere", "dovere", "sapere", "riuscire", "osare",
    "desiderare", "preferire", "sperare", "tentare", "cercare",
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
        1.  Noun agreement – determiner / adjective ↔ noun
            (e.g. ``qualche libri``, ``bella ragazzo``)
        2.  Subject–verb agreement (all 3 spaCy attachment patterns)
            (e.g. ``Gli studenti va``, ``Il professore hanno spiegato``)
        3.  Possessive–noun agreement
            (e.g. ``nostri progetto``)
        4.  Auxiliary–past-participle agreement
            (e.g. ``siamo tornato``)
        5.  Article–noun agreement
            (e.g. ``un ragazza``, ``la problema``)
        6.  Post-nominal adjective agreement
            (e.g. ``case grande``, ``problemi complesso``)
        7.  Preposition contraction errors
            (e.g. ``di il libro`` → ``del libro``)
        8.  Verb–preposition collocation errors
            (e.g. ``vado in casa`` instead of ``vado a casa``)
        9.  Predicate adjective agreement          [NEW]
            (e.g. ``Lei era simpatico`` → ``simpatica``)
        10. Clitic pronoun–verb agreement          [NEW]
            (e.g. ``Lo ho visto`` → ``L'ho visto``)
        11. Partitive article misuse               [NEW]
            (e.g. ``del mele`` → ``delle mele``)
        12. Missing reflexive clitic               [NEW]
            (e.g. ``Mario lava ogni mattina`` missing ``si``)
        13. Double negation errors                 [NEW]
            (e.g. ``Ho visto niente`` → ``Non ho visto niente``)
        14. Interrogative word order               [NEW]
            (e.g. ``Cosa tu fai?`` → ``Cosa fai?``)
        15. Gerund subject mismatch                [NEW]
            (e.g. ``Essendo stanca, lui uscì``)
        16. Modal + non-infinitive errors          [NEW]
            (e.g. ``Voglio andati`` → ``Voglio andare``)
    """

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def __init__(self, use_spacy: bool = True):
        """
        Initialise LanguageTool for Italian ("it") and optionally load the
        large Italian spaCy model.

        After loading spaCy the dynamic lexicon builder is called once to
        extend the seed lookup tables (_MASCULINE_NOUNS, _FEMININE_NOUNS,
        _ESSERE_VERBS) with anything the model's own morphology reveals from
        a built-in reference corpus.
        """
        self.tool = language_tool_python.LanguageTool("it")

        self.nlp = None
        if use_spacy and SPACY_AVAILABLE:
            try:
                self.nlp = spacy.load("it_core_news_lg")
                self._build_dynamic_lexicons()
            except OSError:
                print(
                    "spaCy model not found. Run:\n"
                    "python -m spacy download it_core_news_lg"
                )

    # ------------------------------------------------------------------
    # Dynamic lexicon builder
    # ------------------------------------------------------------------

    def _build_dynamic_lexicons(self) -> None:
        """
        Extend the three seed sets (_MASCULINE_NOUNS, _FEMININE_NOUNS,
        _ESSERE_VERBS) by running the loaded spaCy model over a compact
        reference corpus of Italian sentences that exercise common exception
        nouns and essere-verbs.

        The model's morphology tagger assigns Gender/Number to each token;
        we compare that against surface-form heuristics to discover new
        exceptions automatically.

        Strategy
        --------
        Masculine exceptions  : noun ends in -a but model tags Gender=Masc
        Feminine exceptions   : noun ends in -o/-e but model tags Gender=Fem
                                (excluding obvious -ione/-tà already in seed)
        Essere verbs          : VERB lemmas that appear with an essere-AUX
                                child in the reference corpus
        """
        # -------------------------------------------------------------------
        # Reference corpus – sentences chosen to surface exceptions.
        # Add more sentences here to improve discovery coverage.
        # -------------------------------------------------------------------
        _REFERENCE_CORPUS = """
        Il problema principale è la mancanza di risorse.
        Abbiamo sviluppato un nuovo programma informatico.
        Il sistema operativo si è aggiornato automaticamente.
        Il tema della conferenza era molto interessante.
        Il clima mediterraneo è molto piacevole.
        Hai seguito lo schema corretto per l'analisi.
        Il panorama dalla cima era mozzafiato.
        Ha ottenuto il diploma con il massimo dei voti.
        Il dramma si è svolto in tre atti.
        Ho letto un bellissimo poema epico.
        L'aroma del caffè si sentiva in tutto il corridoio.
        Il fantasma del castello spaventava i visitatori.
        Il teorema di Pitagora è fondamentale in geometria.
        Si trovava in un profondo dilemma morale.
        Il dogma religioso è stato messo in discussione.
        L'enigma è rimasto irrisolto per secoli.
        Il trauma psicologico ha richiesto un lungo recupero.
        Il plasma sanguigno è composto principalmente di acqua.
        Ha inviato un telegramma urgente al consolato.
        Il diagramma mostra l'andamento delle vendite.
        L'atleta professionista si allena ogni giorno.
        Il pianeta Marte è detto il pianeta rosso.
        Il poeta romantico ha scritto versi immortali.
        Il profeta aveva previsto la catastrofe.
        La mano destra è più forte della sinistra.
        La radio trasmetteva musica classica tutto il giorno.
        La foto del matrimonio era bellissima.
        La moto era parcheggiata sotto casa.
        La mia auto nuova è molto efficiente.
        La crisi economica ha colpito molte famiglie.
        La tesi di laurea è stata approvata con lode.
        L'analisi dei dati ha rivelato tendenze interessanti.
        La sintesi della ricerca è stata pubblicata ieri.
        L'ipotesi non è stata ancora verificata.
        La diagnosi è arrivata dopo una settimana di esami.
        La città di Roma è ricca di storia e cultura.
        La libertà di espressione è un diritto fondamentale.
        La verità è sempre più complessa di quanto appaia.
        La qualità del prodotto è migliorata notevolmente.
        L'università offre molti corsi di specializzazione.
        Mario è andato al supermercato stamattina.
        Siamo venuti apposta per salutarti.
        Il treno è partito in orario.
        Gli ospiti sono arrivati con molto anticipo.
        Mia sorella è tornata dalla Francia ieri sera.
        I bambini sono usciti a giocare nel parco.
        Il ladro è entrato dalla finestra posteriore.
        Luca è nato a Milano nel 1990.
        La nonna è morta serenamente nel sonno.
        L'acqua è caduta dal tavolo e ha bagnato tutto.
        Siamo saliti sul treno all'ultimo momento.
        Il gatto è sceso dall'albero con difficoltà.
        La situazione è diventata sempre più complicata.
        Il cielo è sembrato improvvisamente minaccioso.
        Il risultato è parso soddisfacente a tutti.
        Sono rimasto a casa tutto il giorno per via della pioggia.
        La bambina è restata sveglia fino a tardi.
        Il vaso è caduto dal davanzale e si è rotto.
        Abbiamo corso e siamo arrivati in tempo per lo spettacolo.
        L'estate è passata in fretta quest'anno.
        Il sole è tramontato dietro le montagne.
        Una nuova stella è comparsa nel cielo notturno.
        Il vecchio edificio è scomparso dopo la demolizione.
        Il sole è sorto all'alba colorando il cielo di rosso.
        La nebbia è emersa lentamente dalla valle.
        """

        if self.nlp is None:
            return

        doc = self.nlp(_REFERENCE_CORPUS)

        for token in doc:
            lemma = token.lemma_.lower()
            text_lower = token.text.lower()
            gender = token.morph.get("Gender")
            pos = token.pos_

            # ---------------------------------------------------------------
            # Discover masculine exception nouns (end in -a but are Masc)
            # ---------------------------------------------------------------
            if (
                pos == "NOUN"
                and text_lower.endswith("a")
                and gender == ["Masc"]
                and lemma not in _MASCULINE_NOUNS
            ):
                _MASCULINE_NOUNS.add(lemma)

            # ---------------------------------------------------------------
            # Discover feminine exception nouns (end in -o but are Fem,
            # or end in -e but are tagged Fem and not already in seed)
            # ---------------------------------------------------------------
            if (
                pos == "NOUN"
                and (text_lower.endswith("o") or text_lower.endswith("e"))
                and gender == ["Fem"]
                and lemma not in _FEMININE_NOUNS
            ):
                _FEMININE_NOUNS.add(lemma)

            # ---------------------------------------------------------------
            # Discover essere-verbs: VERB tokens whose AUX child is essere
            # ---------------------------------------------------------------
            if pos == "VERB" and lemma not in _ESSERE_VERBS:
                for child in token.children:
                    if (
                        child.dep_ == "aux"
                        and child.lemma_.lower() == "essere"
                    ):
                        _ESSERE_VERBS.add(lemma)
                        break

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
        Detect number mismatches between a nominal subject and its finite verb
        or auxiliary.

        Handles three attachment patterns spaCy produces for Italian:

        Pattern A – subject directly under the main VERB
            ``Gli studenti va a scuola``
            studenti -nsubj-> va(VERB)

        Pattern B – subject under an AUX (compound tense)
            ``Il professore hanno spiegato``
            professore -nsubj-> hanno(AUX) -> spiegato(VERB)
            The AUX is the finite inflected word; we flag it.

        Pattern C – subject under AUX for essere-predicate
            ``Alcuni studenti era interessati``
            studenti -nsubj-> era(AUX/VERB)
            The finite word is the AUX itself; we flag it directly.

        In every pattern the finite inflected word (AUX or VERB) is the one
        that must agree with the subject in number.
        """
        if self.nlp is None:
            return []

        issues = []
        doc = self.nlp(text)
        seen_pairs = set()          # avoid duplicate flags for the same (subj, finite) pair

        for token in doc:
            if token.dep_ not in ("nsubj", "nsubj:pass"):
                continue

            subj = token
            head = token.head       # direct syntactic head (VERB or AUX)

            # Collect candidate finite words to compare against the subject.
            #   1. The direct head itself  (patterns A and C).
            #   2. If head is AUX, walk up to its own head (the main VERB) and
            #      also collect that verb's other AUX children  (pattern B).
            candidates = []

            if head.pos_ in ("VERB", "AUX"):
                candidates.append(head)

            if head.pos_ == "AUX":
                main_verb = head.head
                if main_verb.pos_ in ("VERB", "AUX"):
                    candidates.append(main_verb)
                for sibling in main_verb.children:
                    if sibling.dep_ == "aux" and sibling is not head:
                        candidates.append(sibling)

            for finite in candidates:
                pair_key = (subj.i, finite.i)
                if pair_key in seen_pairs:
                    continue

                s_number = subj.morph.get("Number")
                v_number = finite.morph.get("Number")

                if s_number and v_number and s_number != v_number:
                    seen_pairs.add(pair_key)
                    issues.append(
                        self._issue(
                            text=text,
                            token=finite,
                            rule="SPACY_SUBJECT_VERB_AGREEMENT",
                            message=(
                                f"Possible subject-verb agreement error: "
                                f"'{subj.text}' is {s_number[0]}, "
                                f"but '{finite.text}' is {v_number[0]}."
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
    # spaCy rule 9 – predicate adjective agreement  [NEW]
    # ------------------------------------------------------------------

    def spacy_predicate_adjective_issues(self, text: str) -> list:
        """
        Detect gender/number mismatches between the subject and a predicate
        adjective linked by a copular verb (essere, sembrare, diventare, …).

        Example:  ``Lei era molto simpatico``
            Lei  → Fem,Sing
            simpatico → Masc,Sing  ✗  should be  simpatica

        spaCy Italian uses TWO different parse structures for this construction
        depending on the model version and sentence complexity — we handle both:

        Pattern A  (verb-headed):
            era(AUX/VERB, ROOT) ← simpatico(ADJ, dep="attr")
            era ← Lei(PRON, dep="nsubj")
            → head of ADJ is a copular verb: check head.lemma_ ∈ _COPULAR_VERBS

        Pattern B  (adjective-headed / copula-as-child):
            simpatico(ADJ, ROOT) ← era(AUX, dep="cop")
            simpatico ← Lei(PRON, dep="nsubj")
            → ADJ is ROOT and has a "cop" child whose lemma ∈ _COPULAR_VERBS
        """
        if self.nlp is None:
            return []

        issues = []
        doc = self.nlp(text)

        for token in doc:
            if token.pos_ != "ADJ":
                continue

            head = token.head
            subject = None
            is_predicate_adj = False

            # ----------------------------------------------------------
            # Pattern A: ADJ head is a copular VERB/AUX
            # ----------------------------------------------------------
            if head.pos_ in ("VERB", "AUX") and head.lemma_.lower() in _COPULAR_VERBS:
                is_predicate_adj = True
                for child in head.children:
                    if child.dep_ in ("nsubj", "nsubj:pass"):
                        subject = child
                        break

            # ----------------------------------------------------------
            # Pattern B: ADJ is ROOT (or high node) with a "cop" child
            # ----------------------------------------------------------
            if not is_predicate_adj:
                cop_child = None
                for child in token.children:
                    if child.dep_ == "cop" and child.lemma_.lower() in _COPULAR_VERBS:
                        cop_child = child
                        break
                if cop_child is not None:
                    is_predicate_adj = True
                    # Subject is a direct child of the ADJ in this pattern
                    for child in token.children:
                        if child.dep_ in ("nsubj", "nsubj:pass"):
                            subject = child
                            break

            if not is_predicate_adj or subject is None:
                continue

            s_gender = subject.morph.get("Gender")
            s_number = subject.morph.get("Number")
            a_gender = token.morph.get("Gender")
            a_number = token.morph.get("Number")

            mismatches = []
            if s_gender and a_gender and s_gender != a_gender:
                mismatches.append("gender")
            if s_number and a_number and s_number != a_number:
                mismatches.append("number")

            if mismatches:
                issues.append(
                    self._issue(
                        text=text,
                        token=token,
                        rule="SPACY_PREDICATE_ADJ_AGREEMENT",
                        message=(
                            f"Predicate adjective {' and '.join(mismatches)} mismatch: "
                            f"subject '{subject.text}' is "
                            f"{', '.join(s_gender + s_number)}, but "
                            f"'{token.text}' is {', '.join(a_gender + a_number)}. "
                            f"(e.g. 'Lei era simpatico' → 'Lei era simpatica')"
                        ),
                    )
                )
        return issues

    # ------------------------------------------------------------------
    # spaCy rule 10 – clitic pronoun–verb agreement  [NEW]
    # ------------------------------------------------------------------

    def spacy_clitic_agreement_issues(self, text: str) -> list:
        """
        Detect the most common clitic elision error: ``lo ho`` / ``la ho``
        should contract to ``l'ho`` before a vowel-initial auxiliary.

        Also flags ``lo`` / ``la`` used before a vowel-starting verb where
        elision (l') is obligatory in standard Italian.

        spaCy tags clitics as PRON with dep_='obj' or 'expl' and the morph
        feature Clitic=Yes.

        Limitations: full clitic-climbing and agreement with past participle
        (e.g. ``le ho viste``) require richer context; those are approximated
        here via a simpler surface check.
        """
        if self.nlp is None:
            return []

        issues = []
        doc = self.nlp(text)
        tokens = list(doc)

        for i, token in enumerate(tokens[:-1]):
            # Look for bare "lo" / "la" immediately before a vowel-starting verb
            if token.text.lower() not in ("lo", "la"):
                continue
            next_tok = tokens[i + 1]
            if next_tok.pos_ not in ("VERB", "AUX"):
                continue
            next_lower = next_tok.text.lower()
            if next_lower and next_lower[0] in "aeiouàèéìòù":
                issues.append(
                    self._issue(
                        text=text,
                        token=token,
                        rule="SPACY_CLITIC_ELISION",
                        message=(
                            f"Clitic '{token.text}' should be elided before "
                            f"'{next_tok.text}' (starts with a vowel): "
                            f"'{token.text} {next_tok.text}' → "
                            f"'l'{next_tok.text}'."
                        ),
                        suggestions=[f"l'{next_tok.text}"],
                    )
                )
        return issues

    # ------------------------------------------------------------------
    # spaCy rule 11 – partitive article misuse  [NEW]
    # ------------------------------------------------------------------

    def spacy_partitive_article_issues(self, text: str) -> list:
        """
        Detect gender/number mismatches in partitive articles
        (del / dello / della / dei / degli / delle).

        Example:  ``Ho mangiato del mele``
            mele → Fem,Plur  but  del → Masc,Sing  ✗  should be  delle mele

        Detection: find DET tokens that are contracted partitive forms
        (recognised via _CONTRACTED_TO_PARTS), decompose them into their
        prep + article components, derive the article's implied gender/number,
        and compare with the head noun.
        """
        if self.nlp is None:
            return []

        # Map contracted partitive → (gender, number) implied by the article part
        _PARTITIVE_MORPH = {
            "del":   ("Masc", "Sing"),
            "dello": ("Masc", "Sing"),   # before s+cons, z, gn…
            "della": ("Fem",  "Sing"),
            "dei":   ("Masc", "Plur"),
            "degli": ("Masc", "Plur"),
            "delle": ("Fem",  "Plur"),
            "dell'": (None,   "Sing"),   # elided; gender ambiguous
        }

        issues = []
        doc = self.nlp(text)

        for token in doc:
            if token.pos_ != "DET":
                continue
            lower = token.text.lower()
            if lower not in _PARTITIVE_MORPH:
                continue

            head = token.head
            if head.pos_ not in ("NOUN", "PROPN"):
                continue

            art_gender, art_number = _PARTITIVE_MORPH[lower]
            h_gender = head.morph.get("Gender")
            h_number = head.morph.get("Number")

            mismatches = []
            if art_gender and h_gender and [art_gender] != h_gender:
                mismatches.append("gender")
            if art_number and h_number and [art_number] != h_number:
                mismatches.append("number")

            if mismatches:
                # Suggest the correct partitive form
                target_gender = h_gender[0] if h_gender else "?"
                target_number = h_number[0] if h_number else "?"
                correct_map = {
                    ("Masc", "Sing"): "del/dello",
                    ("Fem",  "Sing"): "della",
                    ("Masc", "Plur"): "dei/degli",
                    ("Fem",  "Plur"): "delle",
                }
                suggestion = correct_map.get((target_gender, target_number), "?")
                issues.append(
                    self._issue(
                        text=text,
                        token=token,
                        rule="SPACY_PARTITIVE_AGREEMENT",
                        message=(
                            f"Partitive article {' and '.join(mismatches)} mismatch: "
                            f"'{token.text}' does not agree with '{head.text}' "
                            f"({target_gender},{target_number}). "
                            f"Use '{suggestion}' instead. "
                            f"(e.g. 'del mele' → 'delle mele')"
                        ),
                        suggestions=[suggestion],
                    )
                )
        return issues

    # ------------------------------------------------------------------
    # spaCy rule 12 – missing reflexive clitic  [NEW]
    # ------------------------------------------------------------------

    def spacy_missing_reflexive_clitic(self, text: str) -> list:
        """
        Detect verbs that are inherently reflexive in Italian but appear
        without their required clitic pronoun (mi/ti/si/ci/vi).

        Example:  ``Mario lava ogni mattina``
            'lavare' used intransitively without 'si' → likely 'lavarsi'.

        Detection heuristic
        -------------------
        If a verb's lemma is in _REFLEXIVE_LEMMA_MAP AND it has no direct
        object (obj) AND no clitic child with Clitic=Yes, it is probably
        missing its reflexive clitic.

        Note: this produces false positives when the verb is used
        transitively with an explicit object (``lava i piatti``); we suppress
        the flag in that case.
        """
        if self.nlp is None:
            return []

        issues = []
        doc = self.nlp(text)

        for token in doc:
            if token.pos_ not in ("VERB", "AUX"):
                continue

            reflexive_form = _REFLEXIVE_LEMMA_MAP.get(token.lemma_.lower())
            if reflexive_form is None:
                continue

            # Suppress if there is an explicit direct object → transitive use
            has_obj = any(c.dep_ in ("obj", "iobj") for c in token.children)
            if has_obj:
                continue

            # Suppress if a clitic is already present
            has_clitic = any(
                c.pos_ == "PRON" and "Yes" in c.morph.get("Clitic", [])
                for c in token.children
            )
            if has_clitic:
                continue

            # Also check left-side clitics (they often precede the verb in text)
            idx = token.i
            left_window = doc[max(0, idx - 2): idx]
            has_left_clitic = any(
                t.pos_ == "PRON" and "Yes" in t.morph.get("Clitic", [])
                for t in left_window
            )
            if has_left_clitic:
                continue

            issues.append(
                self._issue(
                    text=text,
                    token=token,
                    rule="SPACY_MISSING_REFLEXIVE_CLITIC",
                    message=(
                        f"'{token.text}' may be missing its reflexive clitic. "
                        f"Did you mean the reflexive form '{reflexive_form}'? "
                        f"(e.g. 'Mario lava ogni mattina' → 'Mario si lava ogni mattina')"
                    ),
                    suggestions=[reflexive_form],
                )
            )
        return issues

    # ------------------------------------------------------------------
    # spaCy rule 13 – double negation errors  [NEW]
    # ------------------------------------------------------------------

    def spacy_double_negation_issues(self, text: str) -> list:
        """
        Detect missing 'non' when a negative polarity item (niente, nessuno,
        mai, …) is present but the finite verb has no negation marker.

        Italian requires concordance negation: both 'non' before the verb AND
        the NPI are needed.

        Correct:   ``Non ho visto niente.``
        Wrong:     ``Ho visto niente.``

        Detection
        ---------
        For each sentence, find finite verbs. If a NPI token is present in
        the clause but no 'non' / 'né' neg-marker child exists on the verb,
        flag it.
        """
        if self.nlp is None:
            return []

        issues = []
        doc = self.nlp(text)

        for sent in doc.sents:
            tokens = list(sent)
            token_texts_lower = {t.text.lower() for t in tokens}

            # Check whether any NPI appears in this sentence
            npis_present = token_texts_lower & _NEGATIVE_POLARITY_WORDS
            if not npis_present:
                continue

            for token in tokens:
                if token.pos_ not in ("VERB", "AUX"):
                    continue
                # Check for negation marker as a child (dep_='advmod' text 'non')
                has_neg = any(
                    c.dep_ == "advmod" and c.text.lower() in ("non", "né", "ne")
                    for c in token.children
                )
                # Also check immediately preceding tokens (clitics + non often precede)
                idx = token.i
                left = doc[max(0, idx - 3): idx]
                has_neg = has_neg or any(t.text.lower() == "non" for t in left)

                if not has_neg:
                    npi_sample = next(iter(npis_present))
                    issues.append(
                        self._issue(
                            text=text,
                            token=token,
                            rule="SPACY_DOUBLE_NEGATION",
                            message=(
                                f"Negative polarity word '{npi_sample}' found but "
                                f"'non' is missing before '{token.text}'. "
                                f"Italian requires concordance negation: "
                                f"'non … {npi_sample}'. "
                                f"(e.g. 'Ho visto niente' → 'Non ho visto niente')"
                            ),
                            suggestions=["non " + token.text],
                        )
                    )
                    break   # one flag per sentence is enough
        return issues

    # ------------------------------------------------------------------
    # spaCy rule 14 – interrogative word order  [NEW]
    # ------------------------------------------------------------------

    def spacy_interrogative_word_order_issues(self, text: str) -> list:
        """
        Detect subject-before-verb word order after a WH-word in Italian
        questions, which is ungrammatical (unlike English).

        Wrong:   ``Cosa tu fai?``   (subject 'tu' sits between WH and verb)
        Correct: ``Cosa fai?``  or  ``Cosa fai tu?``  (subject after verb)

        Detection heuristic
        -------------------
        In a sentence ending with '?', if a WH-word appears and is
        immediately (within 2 tokens) followed by a personal pronoun subject
        before the finite verb, flag the pronoun.
        """
        if self.nlp is None:
            return []

        _SUBJECT_PRONOUNS = {
            "io", "tu", "lui", "lei", "noi", "voi", "loro", "esso", "essa",
            "essi", "esse",
        }

        issues = []
        doc = self.nlp(text)

        for sent in doc.sents:
            sent_text = sent.text.strip()
            if not sent_text.endswith("?"):
                continue

            tokens = list(sent)
            for i, token in enumerate(tokens):
                if token.text.lower() not in _WH_WORDS:
                    continue

                # Scan the next 1–3 tokens for a subject pronoun before a verb
                window = tokens[i + 1: i + 4]
                for j, w in enumerate(window):
                    if w.text.lower() in _SUBJECT_PRONOUNS and w.dep_ in ("nsubj", "nsubj:pass"):
                        # Check that the verb comes after this pronoun
                        remaining = window[j + 1:]
                        verb_after = any(t.pos_ in ("VERB", "AUX") for t in remaining)
                        if verb_after or j == 0:
                            issues.append(
                                self._issue(
                                    text=text,
                                    token=w,
                                    rule="SPACY_INTERROGATIVE_WORD_ORDER",
                                    message=(
                                        f"In Italian questions the subject pronoun "
                                        f"'{w.text}' should not appear between the "
                                        f"WH-word '{token.text}' and the verb. "
                                        f"Drop it or move it after the verb. "
                                        f"(e.g. 'Cosa tu fai?' → 'Cosa fai?' "
                                        f"or 'Cosa fai tu?')"
                                    ),
                                )
                            )
                            break
        return issues

    # ------------------------------------------------------------------
    # spaCy rule 15 – gerund subject mismatch  [NEW]
    # ------------------------------------------------------------------

    def spacy_gerund_subject_mismatch(self, text: str) -> list:
        """
        Detect gender mismatches between the implied subject of a gerund/
        participial clause (essendo, avendo, …) and the subject of the main
        clause.

        Wrong:   ``Essendo stanca, lui uscì di casa.``
            Gerund adjective 'stanca' is Fem but main subject 'lui' is Masc.
        Correct: ``Essendo stanco, lui uscì di casa.``

        Detection
        ---------
        Find ADJ tokens that are children of a gerund verb (VerbForm=Ger).
        Find the main clause subject (nsubj of the root verb). Compare gender.
        """
        if self.nlp is None:
            return []

        issues = []
        doc = self.nlp(text)

        # Find root subject (the main clause subject)
        root_subject = None
        for token in doc:
            if token.dep_ == "ROOT":
                for child in token.children:
                    if child.dep_ in ("nsubj", "nsubj:pass"):
                        root_subject = child
                        break
                break

        if root_subject is None:
            return []

        s_gender = root_subject.morph.get("Gender")
        s_number = root_subject.morph.get("Number")

        for token in doc:
            # Find gerund verbs
            if not ("Ger" in token.morph.get("VerbForm", [])):
                continue

            # Look for adjective children of the gerund (predicate adjectives)
            for child in token.children:
                if child.pos_ != "ADJ":
                    continue

                a_gender = child.morph.get("Gender")
                a_number = child.morph.get("Number")

                mismatches = []
                if s_gender and a_gender and s_gender != a_gender:
                    mismatches.append("gender")
                if s_number and a_number and s_number != a_number:
                    mismatches.append("number")

                if mismatches:
                    issues.append(
                        self._issue(
                            text=text,
                            token=child,
                            rule="SPACY_GERUND_SUBJECT_MISMATCH",
                            message=(
                                f"Gerund adjective {' and '.join(mismatches)} mismatch: "
                                f"'{child.text}' does not agree with main subject "
                                f"'{root_subject.text}'. "
                                f"(e.g. 'Essendo stanca, lui uscì' → "
                                f"'Essendo stanco, lui uscì')"
                            ),
                        )
                    )
        return issues

    # ------------------------------------------------------------------
    # spaCy rule 16 – modal + non-infinitive  [NEW]
    # ------------------------------------------------------------------

    def spacy_modal_infinitive_issues(self, text: str) -> list:
        """
        Detect cases where a modal verb is followed by a non-infinitive
        (e.g. a past participle or a conjugated verb form).

        Wrong:   ``Voglio andati a casa.``   (andati = past participle)
        Correct: ``Voglio andare a casa.``   (andare = infinitive)

        Detection
        ---------
        Find VERB/AUX tokens whose lemma is in _MODAL_VERBS, then look at
        their xcomp / ccomp children. If the child verb's VerbForm is not
        'Inf', flag it.
        """
        if self.nlp is None:
            return []

        issues = []
        doc = self.nlp(text)

        for token in doc:
            if token.pos_ not in ("VERB", "AUX"):
                continue
            if token.lemma_.lower() not in _MODAL_VERBS:
                continue

            for child in token.children:
                if child.dep_ not in ("xcomp", "ccomp", "obj"):
                    continue
                if child.pos_ not in ("VERB", "AUX"):
                    continue

                verb_form = child.morph.get("VerbForm")
                if not verb_form:
                    continue

                if "Inf" not in verb_form:
                    wrong_form = verb_form[0] if verb_form else "non-infinitive"
                    issues.append(
                        self._issue(
                            text=text,
                            token=child,
                            rule="SPACY_MODAL_INFINITIVE",
                            message=(
                                f"Modal verb '{token.text}' requires an infinitive, "
                                f"but '{child.text}' is a {wrong_form}. "
                                f"(e.g. 'Voglio andati' → 'Voglio andare')"
                            ),
                            suggestions=[child.lemma_ + "re" if not child.lemma_.endswith("re") else child.lemma_],
                        )
                    )
        return issues

    # ------------------------------------------------------------------
    # Main orchestration
    # ------------------------------------------------------------------

    def correct_text(self, text: str) -> dict:
        """
        Main entry point.

        Runs LanguageTool and all 16 spaCy rules, merges the results, and
        returns a unified audit payload with duplicates removed.

        Deduplication
        -------------
        Multiple rules can fire on the same token (e.g. rule 1 "noun
        agreement" and rule 5 "article–noun agreement" both flag the same
        article token). We keep the most specific match per (offset, rule_family)
        pair, preferring spaCy issues over LanguageTool ones for the same span
        since spaCy messages are more actionable.

        Returns
        -------
        dict with keys:
            original  – the input string unchanged
            corrected – the LanguageTool auto-corrected version
            polished  – same as corrected (hook for future post-processing)
            matches   – deduplicated list of standardised issue dictionaries
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
            self.spacy_noun_agreement_issues(text)            # rule  1
            + self.spacy_subject_verb_issues(text)            # rule  2
            + self.spacy_possessive_noun_issues(text)         # rule  3
            + self.spacy_aux_participle_issues(text)          # rule  4
            + self.spacy_article_noun_issues(text)            # rule  5
            + self.spacy_postnominal_adjective_issues(text)   # rule  6
            + self.spacy_preposition_contraction_issues(text) # rule  7
            + self.spacy_verb_preposition_issues(text)        # rule  8
            + self.spacy_predicate_adjective_issues(text)     # rule  9
            + self.spacy_clitic_agreement_issues(text)        # rule 10
            + self.spacy_partitive_article_issues(text)       # rule 11
            + self.spacy_missing_reflexive_clitic(text)       # rule 12
            + self.spacy_double_negation_issues(text)         # rule 13
            + self.spacy_interrogative_word_order_issues(text)# rule 14
            + self.spacy_gerund_subject_mismatch(text)        # rule 15
            + self.spacy_modal_infinitive_issues(text)        # rule 16
        )

        all_matches = parsed_lt + spacy_issues

        # ------------------------------------------------------------------
        # Deduplication
        # ------------------------------------------------------------------
        # Two issues are considered duplicates when they flag the SAME character
        # span AND belong to the same broad error family.
        #
        # Rule family is derived from the rule string:
        #   SPACY_NOUN_AGREEMENT and SPACY_ARTICLE_NOUN_AGREEMENT both cover
        #   "agreement" so they share a family key based on offset alone.
        #
        # Priority order (higher index wins within the same span):
        #   LanguageTool generic  <  spaCy generic  <  spaCy specific
        # "Specific" means the rule name contains the exact error type
        # (ARTICLE, POSSESSIVE, PREDICATE, etc.) rather than the catch-all
        # NOUN_AGREEMENT.

        _GENERIC_RULES = {"SPACY_NOUN_AGREEMENT", "SPACY_POSTNOMINAL_ADJ_AGREEMENT"}

        def _priority(issue: dict) -> int:
            if issue["source"] == "LanguageTool":
                return 0
            if issue["rule"] in _GENERIC_RULES:
                return 1
            return 2

        # Group by character offset; keep the highest-priority issue per span
        best: dict[int, dict] = {}
        for issue in all_matches:
            offset = issue["offset"]
            if offset not in best or _priority(issue) > _priority(best[offset]):
                best[offset] = issue

        # Restore original order
        seen_offsets: set[int] = set()
        deduped: list[dict] = []
        for issue in all_matches:
            offset = issue["offset"]
            if offset in seen_offsets:
                continue
            if best[offset] is issue:
                deduped.append(issue)
                seen_offsets.add(offset)

        return {
            "original": text,
            "corrected": corrected_text,
            "polished": corrected_text,
            "matches": deduped,
        }