class tag_replace:
    def convert_keys_to_lower(self, org_dict):
        return {key.lower(): value for key, value in org_dict.items()}

    def replace_tag(self, sentences, keyword, keyword_dict):
        replaced_sentences = []
        key_tag_lower = keyword.lower()

        for sentence in sentences:
            if key_tag_lower in sentence.lower():
                if key_tag_lower in map(str.lower, keyword_dict.keys()):
                    lst = [sentence]
                    keyword_dict_lower = self.convert_keys_to_lower(keyword_dict)
                    for replacement in keyword_dict_lower.get(key_tag_lower, []):
                        replaced_sentence = sentence.lower().replace(
                            key_tag_lower, replacement
                        )
                        lst.append(replaced_sentence)
                    replaced_sentences.append(lst)

        return replaced_sentences if replaced_sentences else None
