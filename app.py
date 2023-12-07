import streamlit as st

from langchain import LLMChain, PromptTemplate
from langchain.prompts.chat import ChatPromptTemplate

from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chat_models import AzureChatOpenAI
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
from typing import List
import os
from pydantic import BaseModel
from langchain.output_parsers import PydanticOutputParser

import time
import random

# Create a unique base key
base_key = "user_input_text_area"

# Generate a dynamic key using the current timestamp
dynamic_key = f"{base_key}_{int(time.time())}"

import os
US_AZURE_OPENAI_API_KEY = "a4cb1b2e62434109a47a2d8d323b8aae"
US_AZURE_OPENAI_API_TYPE = "azure"
US_AZURE_OPENAI_API_VERSION = "2023-07-01-preview"
US_AZURE_OPENAI_API_BASE = "https://opanaius.openai.azure.com/"
US_AZURE_OPEN_API_DEPLOYMENT_NAME = "hygpt35-16"

llm = AzureChatOpenAI(
                    openai_api_base=US_AZURE_OPENAI_API_BASE,
                    openai_api_version=US_AZURE_OPENAI_API_VERSION,
                    deployment_name=US_AZURE_OPEN_API_DEPLOYMENT_NAME,
                    openai_api_key=US_AZURE_OPENAI_API_KEY,
                    openai_api_type=US_AZURE_OPENAI_API_TYPE,
                    temperature=0,
                )
import json

class OpeningResponse(BaseModel):
    opening_dialouge:str
    options:List[str]
    followup_questions: List[str]
    

HINT_BASED_PROMPT = """
You are a AI radiology professor who wants to have a guiding conversation with residents about the finding of radiology images.

The aim of the professor is to help residents naviagate to the finding

Please start the opening dialogue and provide residents 4-6 with a choice of where he want to lead the conversation. These options act as hints so that residents can start exploration. These options should be the entry points to the exploration
 
Nature of Opening Dialogue: 
   1. Mention you are AI radio proffessor 
   1. Dialogue should include the context of the finding
   2. Dialogue should be pave the way to start a healthy converstation 
   3. Do not directly ask about the diagnosis.
   4. The Dialogue should be objective in nature such that the user chooses the direction to follow the exploration
  
Please Add descriptive follow-up questions for each option. ex why or how 

Nature of Follow-up Question:
    1. The question should require a descriptive answer 
    2. Ask mostly why or how the type of follow-up question 
    3. The question should include the context of the finding


These are the findings: {case_finding}
Please reply only as a format {format_instructions}

Before Replying 
    1. Check its a valid format
   
"""


CONVERSATION_PROMPT = """
You radiology professor and this is a dialog conversation between you and your resident whom you want to help navigate about the RADIOLOGY findings discussed above in the conversation.
This is last Resident's response:  {input}
You only reply your response even if resident response is not related to the context

Qualities Possess by AI Proffessor: 
1. Critical to the RESIDENT response but explain concept very clearly
2. Always listens more and pays attention to the resident response

Style of response of AI Proffessor: 
Response Guidelines
    
    1. Response should consist three parts 
        Part1. Gather your response in these following parts:
            Start: 
                A.Provide small feedack to last response of the Resident
            Middle: 
                Check resident response type:
                    A. if is query ->  Add a comprehesive detail explaining the query 
                    B. if its an answer -> Always Respond if answer is wrong or right. If right greet him if wrong explain the correct answer
                    C. if resident reply is that he dont know -> Gently explain the concept behind the question 
            
            End: Always Add Follow up Question
                A. Keep the conversation going ask him/her with a follow-up question attach your with first respone part using context of both your first response part  and actual finding of the case.
                B. Don't say in your reponse that this is your follow up question, instead gently move on with converstation 


    2. Response Formatting        
        After you gathered all the three parts, Merge all these parts. Response should look like as real radiology proffessor is responding. Dont respond by stating it start, end or middle

\n\nCurrent conversation:\n{history}\nAI Proffessor:

"""


prompt = PromptTemplate(input_variables=['input', 'history'],
                        output_parser=None,
                        partial_variables={}, 
                        template=CONVERSATION_PROMPT,
                        template_format='f-string', validate_template=True)

FEEDBACK_PROMPT = """
This is a linear dialog conversation with AI radio proffesor about the RADIOLOGY findings {case_finding} 

Conversation: {converstation}

REPLY - 
PLEASE provide detailed feedback to the resident about its knowledge about the subject using the above conversation as your reference for the resident performance. REPLY WITH YOUR RESPONSE AS A STRING. Please be gentle and list the areas he can focus to improve further

"""


parser = PydanticOutputParser(pydantic_object=OpeningResponse)

hint_based_prompt = PromptTemplate(
    template=HINT_BASED_PROMPT, 
    input_variables=["case_finding"],
    output_parser = parser,
    partial_variables={"format_instructions": parser.get_format_instructions() })
        
        
converstation_prompt = PromptTemplate(template=CONVERSATION_PROMPT, input_variables=["findings", "converstation"]
        )

feedback_prompt = PromptTemplate(template=FEEDBACK_PROMPT, input_variables=["findings", "converstation"]
        )



# Streamlit app
def main():
    
    
    st.title("Case Finding")
    msgs = StreamlitChatMessageHistory(key="langchain_messages")

    # Text area for the user to paste the MRI report
    case_finding = st.text_area("Paste your report here:", key = 1)

    # Conversation logic        
    
    memory = ConversationBufferMemory(
        human_prefix = "Resident",
        ai_prefix= "Ai Proffessor",
        chat_memory=msgs
        )
    
    conversation = ConversationChain(
    llm=llm,
    memory=memory,
    prompt = prompt,
    verbose = True
    )
    print("Run")
    
    
    def submit():
        user_input = st.session_state.user_input
        msgs.add_user_message(user_input)
        res = conversation(user_input)
        st.session_state.user_input = ''
        
        
    if 'selected_option' not in st.session_state:
        st.session_state.selected_option = None
        
    if "user_input" not in st.session_state:
        st.session_state.user_input = ""
         
    last_message_type = ""
    for index, msg in enumerate(msgs.messages):
        if index == 0:
            opening_message = "Welcome Resident!! This is your AI Proffessor. Thanks for contacting me. I will help you naviagate through this case. To begin I listed few entry points where we can start our exploration"
            st.chat_message("ai").write(opening_message)
        
        elif last_message_type == msg.type:
                continue
            
        else:
            last_message_type = msg.type
            st.chat_message(msg.type).write(msg.content)
               
    # Text area for the user to paste the MRI rep    
    if case_finding:
        if hasattr(st.session_state, 'opening_dialogue') and st.session_state.opening_dialogue:
            print("")
        else:
            chain = LLMChain(llm=llm, prompt=hint_based_prompt, verbose = True)
            res = chain.run(case_finding= case_finding)
            res = json.loads(res)
            opening_dialogue_proffesor = res["opening_dialouge"]
            options = res.get("options")
            options = [str(i) for i in options if i]
            opening_dialogue_proffesor = opening_dialogue_proffesor + 'These are following directions in which we can explore the findings'+ ''.join(options)
            msgs.add_ai_message(opening_dialogue_proffesor)
            st.session_state.opening_dialogue = True
            st.session_state.opening_response = res
            
            st.experimental_rerun()
    
    if hasattr(st.session_state, 'opening_response') and st.session_state.opening_response:
        if hasattr(st.session_state, 'selected_option') and st.session_state.selected_option:
            print("")
        else:
            options = st.session_state.opening_response.get("options")
            options = [str(i) for i in options if i]
            options.insert(0, "")
    
            selected_option = st.selectbox(label = "Select the option", options = options)
            options = [i for i in options if i]
            st.session_state.selected_option = selected_option
            if selected_option:
        
                msgs.add_user_message(selected_option)
                follup_question = st.session_state.opening_response.get("followup_questions")
                
                selected_question_index = options.index(selected_option)
                
                if follup_question:
                    msgs.add_ai_message(follup_question[selected_question_index])
                else:
                    msgs.add_ai_message("What you want to know further about this?")
    
                st.session_state.selected_option = selected_option
                st.experimental_rerun()
            
    
    
    if hasattr(st.session_state, 'selected_option') and  st.session_state.selected_option: 
        user_input = st.text_area("Enter text:", key = "user_input", on_change=submit)
    
            
if __name__ == "__main__":
    main()


 
