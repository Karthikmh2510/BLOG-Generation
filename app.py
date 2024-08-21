import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_community.llms import CTransformers

## Func to get response from llama2 model
def getllamaresponse(input_text, no_words,blog_style):
    # Let's call our downloaded LLama2 model
    llm = CTransformers(
        model='K:\LC_Projects\BLOG_Generation\models\llama-2-7b-chat.ggmlv3.q8_0.bin',
        model_type='llama',
        config={'max_new_tokens':256,
                'temperature':0.01})
    
    #Prompt Template
    template = """
    Write a blog for {blog_style} job profile for 
    a topic {input_text}
    within {no_words} words.
    """

    prompt= PromptTemplate(input_variables=["blog_style","input_text","no_words"],
                           template=template)
    
    # Generate Response form LLama 2 model
    response = llm.invoke(prompt.format(blog_style=blog_style,
                       input_text=input_text,
                       no_words=no_words))
    print(response)
    return response

#Let's set our chatbot page config
st.set_page_config(page_title="Generate Blogs",
                   page_icon="🤖",
                   layout='centered',
                   initial_sidebar_state='collapsed')

st.header("Generate Blogs 🤖")

input_text = st.text_input("Enter the Blog topic")

## Create 2 more columns for addition 2 fields
col1, col2 = st.columns([5,5])

with col1:
    no_words = st.text_input("Num of Words")
with col2:
    blog_style=st.selectbox('Writing the blog for..',
                            ('Researchers','Data Scientist','Common People'),
                            index=0)
submit=st.button("Generate")

## Final Response
if submit:
    st.write(getllamaresponse(input_text, 
                              no_words,
                              blog_style))