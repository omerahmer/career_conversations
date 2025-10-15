import json
from openai import OpenAI
from me.tools import tools, record_user_details, record_unknown_question
from me.evaluator import Evaluator
from me.rag import RAG

class Me:
    def __init__(self):
        self.openai = OpenAI()
        self.name = "Syed Omer Ahmer"
        self.rag = RAG()
        with open("me/data/summary.txt", "r", encoding="utf-8") as f:
            self.summary = f.read()
        self.evaluator = Evaluator(self.name, self.summary, "")

    def handle_tool_call(self, tool_calls):
        results = []
        for tool_call in tool_calls:
            tool_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)
            print(f"Tool called: {tool_name}")
            if tool_name == "record_user_details":
                result = record_user_details(**arguments)
            elif tool_name == "record_unknown_question":
                result = record_unknown_question(**arguments)
            else:
                result = {}
            results.append({"role": "tool", "content": json.dumps(result), "tool_call_id": tool_call.id})
        return results

    def system_prompt(self, retrieved_context):
        return f"""
You are acting as {self.name}.
You represent {self.name} professionally on his website, answering questions about his career, skills, and background.
If you don't know something, use the record_unknown_question tool.
Encourage interested users to share their email (record_user_details tool).

## Context (retrieved from files)
{retrieved_context}
"""

    def rerun(self, reply, message, history, feedback):
        updated_prompt = self.system_prompt("") + f"\nThe previous response was rejected. Feedback: {feedback}\n"
        messages = [{"role": "system", "content": updated_prompt}] + history + [{"role": "user", "content": message}]
        response = self.openai.chat.completions.create(model="gpt-4o-mini", messages=messages)
        return response.choices[0].message.content

    def chat(self, message, history):
        retrieved = self.rag.retrieve(message)
        context = "\n---\n".join(retrieved)
        messages = [{"role": "system", "content": self.system_prompt(context)}] + history + [{"role": "user", "content": message}]

        done = False
        while not done:
            response = self.openai.chat.completions.create(model="gpt-4o-mini", messages=messages, tools=tools)
            msg = response.choices[0].message
            if response.choices[0].finish_reason == "tool_calls":
                tool_calls = msg.tool_calls
                results = self.handle_tool_call(tool_calls)
                messages.append(msg)
                messages.extend(results)
            else:
                done = True

        reply = response.choices[0].message.content
        evaluation = self.evaluator.evaluate(reply, message, history)

        if not evaluation.is_acceptable:
            reply = self.rerun(reply, message, history, evaluation.feedback)
        return reply
