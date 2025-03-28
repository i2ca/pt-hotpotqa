

class HotpotEntryValidator():

    def validate(hotpot_entry):
        error_messages = []

        if "question" not in hotpot_entry:
            error_messages.append("Missing question field")
        if "answer" not in hotpot_entry:
            error_messages.append("Missing answer field")
        if "supporting_facts" not in hotpot_entry:
            error_messages.append("Missing supporting_facts field")
        else:
            if "title" not in hotpot_entry['supporting_facts']:
                error_messages.append("Missing supporting_facts.title field")
        if "context" not in hotpot_entry:
            error_messages.append("Missing context field")
        else:
            if "title" not in hotpot_entry['context']:
                error_messages.append("Missing context.title field")
            if "sentences" not in hotpot_entry['context']:
                error_messages.append("Missing context.sentences field")
        return error_messages