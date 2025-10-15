from openai import OpenAI
from dataclasses import dataclass

client = OpenAI()

@dataclass
class Evaluation:
    is_acceptable: bool
    feedback: str

class Evaluator:
    def __init__(self, name, summary, linkedin):
        self.name = name
        self.summary = summary
        self.linkedin = linkedin

    def evaluate(self, reply, message, history):
        system_prompt = f"""You are an evaluation model checking whether the following reply is an accurate, professional, and faithful representation of {self.name}'s career background.

You have access to their summary and LinkedIn profile text:

SUMMARY:
{self.summary}

LINKEDIN:
{self.linkedin}

Evaluate the given reply for these criteria:
1. Faithfulness: Does it reflect {self.name}'s actual experience, skills, and projects?
2. Professional tone: Is it written clearly and respectfully, suitable for a potential employer or client?
3. Relevance: Does it answer the user's question appropriately?

If it fails on any of these, explain why and suggest improvement areas."""

        user_prompt = f"User question: {message}\n\nProposed reply: {reply}\n\nProvide structured feedback."

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )

        feedback = response.choices[0].message.content
        acceptable = not any(word in feedback.lower() for word in ["inaccurate", "unprofessional", "irrelevant", "wrong", "needs improvement"])
        return Evaluation(is_acceptable=acceptable, feedback=feedback)
