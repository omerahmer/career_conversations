from agent import Me
import gradio as gr

if __name__ == "__main__":
    me = Me()
    gr.ChatInterface(me.chat, type="messages", title="Syed Omer Ahmer – AI Assistant").launch()