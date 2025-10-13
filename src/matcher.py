from spacy.language import Doc
from spacy.tokens import Span
from dataclasses import dataclass
from typing import Dict, Tuple, Iterable, List
import re


@dataclass
class Match:
    pattern_name: str
    matched: str
    sentence: str
    
    def __repr__(self) -> str:
        return f"{self.pattern_name} : {self.sentence}"


class SyntaxRegexMatcher:
    """
    This class encapsulates the sentence regex patterns and methods to apply them to target documents.
    Based on the original gram2vec implementation.
    """
    
    def __init__(self, language: str = "en"):
        if language == "en":
            print("Using English constructions")
            AP = r"(?:\([^-]*-[^-]*-JJ-amod(?:\([^-]*-[^-]*-RB-advmod\))*\))*"
            NP = r"[^-]*-[^-]*-NN-[^-]*" + r"(?:\([^-]*-[^-]*-DT-det\))?" + AP
            NPsubj = r"[^-]*-[^-]*-NN-nsubj" + r"(?:\([^-]*-[^-]*-DT-det\))?" + AP
            Nsubj = r"(?:[^-]*-[^-]*-PRP-nsubj|" + r"[^-]*-[^-]*-NN-nsubj" + r"(?:\([^-]*-[^-]*-DT-det\))?" + AP + r")"
            Ndobj = r"(?:[^-]*-[^-]*-PRP-dobj|" + r"[^-]*-[^-]*-NN-dobj" + r"(?:\([^-]*-[^-]*-DT-det\))?" + AP + r")"
            
            self.patterns = {
                # Eric Constructions
                "it-cleft": r"\([^-]*-be-[^-]*-[^-]*.*\([iI]t-it-PRP-nsubj\).*\([^-]*-[^-]*-NN[^-]*-attr.*\([^-]*-[^-]*-VB[^-]*-(relcl|advcl)",
                "pseudo-cleft": r"\([^-]*-be-[^-]*-[^-]*.*\([^-]*-[^-]*-(WP|WRB)-(dobj|advmod)",
                "all-cleft": r"(\([^-]*-be-[^-]*-[^-]*.*\([^-]*-all-(P)?DT)|(\([^-]*-all-(P)?DT-[^-]*.*\([^-]*-be-[^-]*)",
                "there-cleft": r"\([^-]*-be-[^-]*-[^-]*.*\([^-]*-there-EX-expl.*\([^-]*-[^-]*-[^-]*-attr.*\([^-]*-[^-]*-[^-]*-(relcl|acl)",
                "if-because-cleft": r"\([^-]*-be-[^-]*-[^-]*.*\([^-]*-[^-]*-[^-]*-advcl\([^-*]*-if-IN-mark",
                "passive": r"\([^-]*-[^-]*-(NN[^-]*|PRP|WDT)-nsubjpass.*\([^-]*-be-[^-]*-auxpass",
                "subj-relcl": r"\([^-]*-[^-]*-[^-]*-relcl.*\([^-]*-[^-]*-(WP|WDT)-nsubj",
                "obj-relcl": r"\([^-]*-[^-]*-NN[^-]*-(nsubj|attr).*\([^-]*-[^-]*-[^-]*-(relcl|ccomp).*\([^-]*-[^-]*-(WP|WDT|IN)-(pobj|dobj)",
                "tag-question": r"\([^-]*-(do|be|could|can|have)-[^-]*-ROOT.*\(\?-\?-\.-punct",
                "coordinate-clause": r"\([^-]*-[^-]*-CC-cc\).*\([^-]*-[^-]*-(VB[^-]*|JJ)-conj.*\([^-]*-[^-]*-[^-]*-nsubj",
                # Elizabeth Constructions
                # "acc-conjoined-subj" : r"(?:[Mm]e|[Hh]er|[Hh]im|)^([Hh]e)^[Ss]he^I^(-)-[^-]*-PRP-nsubj|[^-]*-[^-]*-[^-]*-nsubj\([^-]*-[^-]*-CC-cc\)\((?:[Mm]e|[Hh]er|[Hh]im)-[^-]*-PRP-conj",
                # "nom-conjoined-obj" : r"\((^(him)^(her)^(me)^(-))-[^-]*-PRP-dobj|[^-]*-[^-]*-NNP-dobj\([^-]*-[^-]*-CC-cc\)\(?:^(him)^(her)^(me)^(-)-[^-]*-PRP-conj|[^-]*-[^-]*-NNP-conj",
                # "adj-amod" : r"\([^-]*-[^-]*-(?:JJ|RB)-[^-]*\([^-]*-[^-]*-RB-advmod\)",
                # "adj-vmod" : r"[^-]*-[^-]*-VB[^-]*-[^-]*(\([^-]*-[^-]*-[^-]*-(?:nsubj|dobj|iobj|advmod))\)*\([^-]*-[^-]*-JJ-acomp|advmod\)",
                # "center-embedding" : r"[^-]*-[^-]*-VB[^-]*-[^-]*\(" + Nsubj + r"\([^-]*-[^-]*-VB[^-]*-relcl(?:[^-]*-[^-]*-ADP-prep)?(?:\([^-]*-[^-]*-WDT-[dp]obj\))?\)?" + Nsubj,
                # "neg-inversion" : r"[^-]*-[^-]*-VB[^-]?-ROOT(?:\(then-then-RB-advmod\([oO]nly-only-RB-advmod\)|\([^-]*-[^-]*-VB[^-]?-advcl\([Oo]nly-only-RB-advmod\)\([^-]*-[^-]*-IN-mark\)\("+ Nsubj + "\)|\((?:[Rr]arely|[Ss]eldom)-[^-]*-RB-advmod\)|\(again-again-RB-advmod\([Nn]ever-never-RB-neg\))\([^-]*-[^-]*-[^-]*-aux\)", 
                # "simple-vp-ellip" : r"\([^-]*-[^-]*-VB[^-]?-ROOT(?:\([^-]*-[^-]*-(?:^VB[^-]?|[^-])*-[^-]*)*(?:\([^-]*-(((?:do|wo|did|should|shall|can|could|will|would|have|has|had|does|may|might|must|am|is|was|were|be|are|being|been)-(?:MD|VB[^-]?)-(?:punct|conj|advcl|relcl))|(to-to-TO-[^-]*))|(?:\(either|neither-either|neither-RB-conj))",
                # "matrix-vp-ellip" : r"\(to-to-TO-xcomp\)?\([^-]*-[^-]*-VB[^-]?-advcl|\((?:do|did|should|shall|can|could|will|wo|would|have|has|had|does|may|might|must|am|is|was|were|be|are|being|been)-[^-]*-(?:MD|VB[^-]?)-ROOT(?:\(\S*\))*\([^-]*-[^-]*-VB[^-]?-advcl",
                # "subord-vp-ellip" : r"advcl\(to-to-TO-xcomp|\([^-]*-[^-]*-VB[^-]?-ROOT(?:\(\S*\))*(?:\(to-to-TO-xcomp|\((?:do|did|should|shall|can|could|will|wo|would|have|has|had|does|may|might|must|am|is|was|were|be|are|being|been)-[^-]*-(?:MD|VB[^-]?)-advcl)",
                # # Daniel Constructions
                # "needs_washed" : r"\([^-]*-(want.*|need.*|like.*)-VB.*-ROOT([^-]*-[^-]*-[^-]*-nsubj.*([^-]*-[^-]*-VBN-xcomp))\)|\([^-]*-[^-]*-VBN-ROOT.*\([^-]*-(need.*|want.*|like.*)-VB.*",
                # "subject_control" : r"\([^-]*-(want|manage|try|decide)-[^-]*-[^-]*.*\([^-]*-[^-]*-[^-]*-nsubj\).*\([^-]*-[^-]*-VB-xcomp.*\(to-to-TO-aux\)",
                # "subject_raising" : r"\([^-]*-(seem|appear|tend|happen)-[^-]*-[^-]*.*\([^-]*-[^-]*-[^-]*-nsubj\).*\([^-]*-[^-]*-VB-xcomp.*\(to-to-TO-aux\)",
                # "personal_dative" : r"\([^-]*-[^-]*-VB.-ROOT([^-]*-[^-]*-[^-]*-[^-]*)\([^-]*-[^-]*-PRP-dative\)",
                # "object_topicalization" : r"\([^-]*-[^-]*-[^-]*-ROOT.*\([^-]*-[^-]*-[^-]*-[^-]*\)*\([^-]*-[^-]*-[^-]*-nsubj\)",
                # "neither_nor" : r"\([^-]*-[^-]*-[^-]*-ROOT\(Neither-neither-CC-preconj\).*\(nor-nor-CC-cc\)",
                # "third_person_subjectless" : r"\([^-]*-[^-]*-VBZ-ROOT\([^-]*-[^-]*-[^-]*-((?!nsubj).)*\)",
                # "tough_movement" : r"nsubj\)\((tough|hard|easy|tricky|difficult)-(tough|hard|easy|tricky|difficult)-JJ-acomp\([^-]*-[^-]*-VB-xcomp\(to-to-TO-aux\)",
                # "imperative_subjectless" : r"\([^-]*-[^-]*-VB-ROOT\([^-]*-[^-]*-[^-]*-((?!nsubj).)*\)",
                # "done_dinner" : r"\((done|finished|started)-[^-]*-[^-]*-ROOT([^-]*-[^-]*-[^-]*-nsubj.*)([^-]*-be-VBP-auxpass)\([^-]*-[^-]*-[^-]*-dobj\)"
<<<<<<< HEAD

                # Hannah Construction
                "parenthetical": r"\([^-]*-[^-]*-[^-]*-ROOT(?!(.*-VB.*))*\([^-]*-[^-]*-[^-]*-nsubj(?!(.*-VB.*))*\([^-]*-[^-]*-[^-]*-punct\).*\([^-]*-[^-]*-[^-]*-[^)]*\([^-]*-[^-]*-[^-]*-(appos|advmod|mark|prep).*\([^-]*-[^-]*-[^-]*-punct\)"
=======
>>>>>>> main
            } 
        elif language == "ru":
            # print("using russian constructions")
            self.patterns = {
                "passive_rus" : r"\([^-]*-[^-]*-VERB-ROOT.*?Voice=Pass.*?\)+$",
                "parataxis_rus": r"\([^-]*-[^-]*-[^-]*-parataxis.*\)",
                "participle_rus": r"\([^-]*-[^-]*-(?:VERB|ADJ)-ROOT.*?(?:ADJ|VERB)-(?:amod|acl)(?!:relcl).*?(?:VerbForm=Part|[а-я]+(?:ющ|ащ|ящ|вш|ем|им|нн|т)[а-я]+).*?\)+$",
                "gerund_rus": r"\([^-]*-[^-]*-(?:VERB|PROPN)-ROOT.*?(?:[А-Яа-я]+(?<!ющ|[ая]щ|вш)[яв(сь|ся)]-[^-]*-(?:VERB|NOUN|ADJ|PROPN)-(?:advcl|amod).*?(?:VerbForm=Conv|[^|]*)).*?\)+$",
                "conj_rus": r"\([^-]*-[^-]*-(?!SCONJ)\w+-ROOT.*?-conj.*?\)+$",
                "nested_structure_rus": r"\([^-]*-[^-]*-(?:VERB|ADV|NOUN)-ROOT.*?(?:VERB-(?:xcomp|advcl|ccomp|acl:relcl|acl)|AUX-aux:pass|ADV-advcl).*?(?:SCONJ-(?:mark|fixed|obj)|ADV-(?:mark|advmod)|который-PRON|что-PRON|чтобы-SCONJ|хотя-SCONJ|такой-DET).*?\)+$",
                "one_word_sent_rus": r"\([^-]*-[^-]*-(NOUN|VERB|PROPN)-ROOT-[^()]*\([^-]*-[^-]*-PUNCT-punct-\)\)",
                "diminutive_rus": r".*\([а-яё]*(?:ик|[её]к|[её]нок|очк|ечк|ышк|оньк|еньк)[а-яё]*-[а-яё]*-NOUN-.*?\).*",
                "multiple_punct_rus": r"\([^-]*-[^-]*-[^-]*-ROOT.*?(?:(?:\(([!?])\1*-\1\1*-PUNCT-punct-\)){2,}|\(\.{2,}-\.{2,}-PUNCT-punct-\))\)+$", # universal
                "additional_info_rus": r"\([^-]*-[^-]*-[^-]*-ROOT.*?(?:\([^-]*\([^)]+\)[^-]*-[^-]*-PUNCT-punct-[^)]*\)|(?:\([^-]*-[^-]*-PUNCT-punct-[^)]*\))){2,}\)+$" #universal
            }
        else:
            raise ValueError(f"Unsupported language: {language}")
        
        self.language = language

    def print_patterns(self) -> None:
        """Print all registered patterns for debugging."""
        for pattern_name, pattern in self.patterns.items():
            print(f"{pattern_name} : {pattern}\n")

    def _find_treegex_matches(self, doc: Doc) -> Tuple[Match]:
        """Iterates through a document's sentences, applying every regex to each sentence."""
        matches = []
        for sent in doc.sents:
            tree_string = self.linearize_tree(sent)
            for name, pattern in self.patterns.items():
                match = re.search(pattern, tree_string)
                if match:
                    matches.append(Match(name, match.group(), sent.text))
        return tuple(matches)

    def add_patterns(self, patterns: Dict[str, str]) -> None:
        """Updates the default patterns dictionary with user-supplied patterns."""
        self.patterns.update(patterns)
        
    def remove_patterns(self, to_remove: Iterable[str]) -> None:
        """Given an iterable of pattern names, removes those patterns from the registered patterns list."""
        for pattern_name in to_remove:
            try:
                del self.patterns[pattern_name]
            except KeyError:
                raise KeyError(f"Pattern '{pattern_name}' not in registered patterns.")
            
    def match_document(self, document: Doc) -> Tuple[Match]:
        """
        Applies all registered patterns to one spaCy-generated document.
        
        Args:
            document: a single spaCy document
            
        Returns:
            a tuple of sentence matches for a single document
        """
        return self._find_treegex_matches(document)

    def match_documents(self, documents: Iterable[Doc]) -> List[Tuple[Match]]:
        """
        Applies all registered patterns to a collection of spaCy-generated documents.
        
        Args:
            documents: iterable of spaCy documents
            
        Returns:
            A list of tuples such that each tuple contains one document's sentence matches
        """
        all_matches = []
        for document in documents:
            all_matches.append(self._find_treegex_matches(document))
        return all_matches
    
    def linearize_tree(self, sentence: Span) -> str:
        """Converts a spaCy dependency-parsed sentence into a linear tree string while preserving dependency relations."""
        
        def get_NT_count(sentence) -> int:
            """Returns the number of non-terminal nodes in a dep tree."""
            return sum([1 for token in sentence if list(token.children)])

        def ending_parenthesis(n: int) -> str:
            """Returns the appropriate amount of parenthesis to add to linearized tree."""
            return f"{')' * n}"
        
        def parse_dependency_parse(sentence):
            """Processes a dependency parse in a bottom-up fashion."""
            stack = [sentence.root]
            result = ""
            while stack:
                token = stack.pop()
                result += f"({token.text}-{token.lemma_}-{token.tag_}-{token.dep_}" 
                
                for child in reversed(list(token.children)):
                    stack.append(child)
                
                if not list(token.children):
                    result += ")"
            return result
        
        parse = parse_dependency_parse(sentence)
        nt_count = get_NT_count(sentence)
        return f"{parse}{ending_parenthesis(nt_count)}"
