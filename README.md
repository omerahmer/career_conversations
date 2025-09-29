# career_conversations

# 💬 Personal Website AI Chatbot

This project implements a conversational agent that acts as me (**Syed Omer Ahmer**) on my website. The agent uses **OpenAI** and **Gemini** APIs to answer questions professionally and engagingly, as if talking to a potential client or employer.  

It includes:  
- Context from LinkedIn and a written summary.  
- Automatic evaluation of responses for quality control.  
- Retry mechanism with improved answers if evaluation fails.  
- Tools to **record user contact details** and **log unanswered questions**.  
- Push notifications via **Pushover** for important events.  
- A **Gradio chat interface** for testing locally.  

---

## Features

- **Persona Simulation**  
  Represents my career, skills, and experience in a natural, professional way.  

- **Response Evaluation**  
  Every agent reply is checked by Gemini (`gemini-2.0-flash`) using a Pydantic model (`Evaluation`). If rejected, the agent retries with improved wording.  

- **Tool Use**  
  - `record_user_details`: Logs when a visitor provides an email/name/notes.  
  - `record_unknown_question`: Records any question the agent can’t answer.  
  Both trigger **Pushover notifications**.  

- **Notifications**  
  Important actions (user details, unknown questions) are instantly pushed to your phone via Pushover.  

- **Gradio Interface**  
  Provides a simple UI to chat with the agent during development.  

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/your-username/personal-website-chatbot.git
cd personal-website-chatbot
```

### 2. Install dependencies
```bash
uv sync
```

### 3. Setup environment variables
Create a .env file in the project root
```bash
OPENAI_API_KEY=your_openai_api_key
GEMINI_API_KEY=your_gemini_api_key
PUSHOVER_TOKEN=your_pushover_app_token
PUSHOVER_USER=your_pushover_user_key
```

### 4. Prepare your data
Save your LinkedIn profile PDF as me/linkedin.pdf.
Write a summary of your background in me/summary.txt.

### 5. Run locally