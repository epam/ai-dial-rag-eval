from match_facts import canonize


def highlight_context_by_facts(row=None, facts_column="ground_truth_facts", context_column="context"):
    facts = row[facts_column]
    contexts = row[context_column]

    highlighted_contexts = []
    for context in contexts:
        for fact in facts:
            if fact in context:
                context = context.replace(
                    fact, f'<span style="background-color: yellow;">{fact}</span>'
                )
        highlighted_contexts.append(context)
    list_items = "".join(f"<li>{context}</li>" for context in highlighted_contexts)
    final_html_list = f"<ol>{list_items}</ol>"
    return final_html_list


def highlight_facts_by_context(row):
    facts = row["ground_truth_facts"]
    contexts = row["context"]

    highlighted_facts = []
    for fact in facts:
        for context in contexts:
            if fact in context:
                fact = f'<span style="background-color: yellow;">{fact}</span>'
        highlighted_facts.append(fact)
    list_items = "".join(f"<li>{fact}</li>" for fact in highlighted_facts)
    final_html_list = f"<ol>{list_items}</ol>"
    return final_html_list


def highlight_details(eval_data, canonize_strings=False):
    eval_data_highlighted = eval_data.copy()
    eval_data_canonized = eval_data.copy()
    if canonize_strings:
        eval_data_canonized["context"] = eval_data["context"].apply(
            lambda x: [canonize(s) for s in x]
        )
        eval_data_canonized["ground_truth_facts"] = eval_data[
            "ground_truth_facts"
        ].apply(lambda x: [canonize(s) for s in x])

    eval_data_highlighted["context"] = eval_data_canonized.apply(
        highlight_context_by_facts, axis=1
    )
    eval_data_highlighted["ground_truth_facts"] = eval_data_canonized.apply(
        highlight_facts_by_context, axis=1
    )
    return (
        eval_data_highlighted[["question", "ground_truth_facts", "context"]]
        .style.set_properties(width="5%", **{"text-align": "left", "escape": False})
        .set_properties(subset=["question"], width="10%")
        .set_properties(subset=["ground_truth_facts"], width="25%")
        .set_properties(subset=["context"], width="65%")
    )
