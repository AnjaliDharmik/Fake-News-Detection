import streamlit as st
import logging
import trafilatura
import plotly.graph_objects as go

import tensorflow as tf
import pickle
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def predict_fake_news(text: str):
    # Load model
    model = tf.keras.models.load_model("Saved_Model/Fake_News_Detector_model.h5")

    # Load vectorizer and selector
    with open("Saved_Model/vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)

    with open("Saved_Model/selector.pkl", "rb") as f:
        selector = pickle.load(f)

    # Preprocess input
    x_input = vectorizer.transform([text])
    x_input = selector.transform(x_input).astype("float32")
    x_input = x_input.toarray()

    # Make prediction
    prediction_prob = model.predict(x_input)[0][0]
    prediction = "Real News" if prediction_prob >= 0.5 else "Fake News"
    confidence = prediction_prob if prediction_prob >= 0.5 else 1 - prediction_prob

    print(f"\nPrediction: {prediction}")
    print(f"Confidence: {confidence:.2f}")
        
    return prediction, confidence

def display_prediction_results(prediction, confidence, article_text):
    """Display prediction results with styling"""
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Main result
        if prediction == 1:
            st.success("✅ **REAL NEWS**")
            result_color = "green"
        else:
            st.error("❌ **FAKE NEWS**")
            result_color = "red"
        
        # Confidence score
        st.metric(
            label="Confidence Score",
            value=f"{confidence:.2%}",
            help="How confident the model is in its prediction"
        )
        
        # Confidence gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=confidence * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Confidence Level (%)"},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': result_color},
                'steps': [
                    {'range': [0, 50], 'color': "lightgray"},
                    {'range': [50, 80], 'color': "yellow"},
                    {'range': [80, 100], 'color': "lightgreen"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Article statistics
        st.subheader("📊 Article Statistics")
        word_count = len(article_text.split())
        char_count = len(article_text)
        
        st.metric("Word Count", word_count)
        st.metric("Character Count", char_count)
        
        # Risk assessment
        st.subheader("🎯 Risk Assessment")
        if confidence > 0.9:
            risk_level = "Very High Confidence"
            risk_color = "green" if prediction == 1 else "red"
        elif confidence > 0.7:
            risk_level = "High Confidence"
            risk_color = "orange"
        else:
            risk_level = "Low Confidence"
            risk_color = "yellow"
        
        st.markdown(f"**Risk Level:** <span style='color: {risk_color}'>{risk_level}</span>", 
                   unsafe_allow_html=True)

if 'extracted_article' not in st.session_state:
    st.session_state.extracted_article = ""

def extract_article_from_url(url):
    """Extract article content from a given URL using trafilatura"""
    try:
        # Download the webpage
        downloaded = trafilatura.fetch_url(url)
        
        if not downloaded:
            return None
        
        # Extract the main text content
        text = trafilatura.extract(downloaded)
        
        return text
        
    except Exception as e:
        logger.error(f"Error extracting article from URL {url}: {str(e)}")
        raise e

def main():
    st.title("🔍 Fake News Detection System")
    st.markdown("---")

    st.header("📰 News Article Analysis")
    
    # Input methods
    input_method = st.radio(
        "Choose input method:",
        ["Type/Paste Article", "Extract from URL"]
    )
    
    article_text = ""
    
    if input_method == "Type/Paste Article":
        article_text = st.text_area(
            "Enter news article text:",
            height=200,
            placeholder="Paste the news article content here..."
        )

    elif input_method == "Extract from URL":
        st.info("🌐 **URL Extraction Feature**: Enter a news article URL to automatically extract and analyze the content.")
        
        # URL input with helpful examples
        url = st.text_input(
            "Enter news article URL:",
            placeholder="https://example.com/news-article"
        )
        
        # Show example URLs
        with st.expander("📋 Example URLs to try"):
            st.markdown("""
            You can try these types of news sources:
            - **Major News Sites**: CNN, BBC, Reuters, Associated Press
            - **Newspapers**: New York Times, Washington Post, Guardian
            - **Technology News**: TechCrunch, Wired, Ars Technica
            - **Science News**: Nature, Scientific American, Science Daily
            
            **Note**: Some sites may require subscription or have anti-bot protection.
            For best results, use publicly accessible articles.
            """)
        
        if url and st.button("🌐 Extract Article from URL"):
            try:
                with st.spinner("Extracting article content from URL..."):
                    extracted_text = extract_article_from_url(url)
                
                if extracted_text and len(extracted_text.strip()) > 100:
                    st.success("✅ Article extracted successfully!")
                    
                    # Store in session state
                    st.session_state.extracted_article = extracted_text
                    
                    # Show extracted content statistics
                    word_count = len(extracted_text.split())
                    char_count = len(extracted_text)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Words Extracted", word_count)
                    with col2:
                        st.metric("Characters Extracted", char_count)
                    
                    st.text_area("Extracted content:", value=extracted_text, height=200, disabled=True)
                    article_text = extracted_text
                else:
                    st.error("❌ Failed to extract meaningful content. This could be due to:")
                    st.markdown("""
                    - **Paywall or login required** - Try a different article
                    - **Anti-bot protection** - Some sites block automated access
                    - **Invalid URL format** - Check the URL is correct
                    - **Content not accessible** - Try a publicly available article
                    """)
                    
            except Exception as e:
                st.error(f"❌ Error extracting article: {str(e)}")
                st.markdown("**Troubleshooting tips:**")
                st.markdown("- Ensure the URL is valid and accessible")
                st.markdown("- Try a different news source")
                st.markdown("- Check your internet connection")
        
        # Use extracted article if available
        if st.session_state.extracted_article:
            article_text = st.session_state.extracted_article
            if st.button("🗑️ Clear Extracted Article"):
                st.session_state.extracted_article = ""
                st.rerun()

    
    # Analysis button
    if st.button("🔍 Analyze Article", type="primary"):
        if not article_text.strip():
            st.error("Please paste or enter url of an article to analyze.")
            return
        
        try:
            with st.spinner("Analyzing article..."):
                prediction, confidence = predict_fake_news(article_text)
            
            display_prediction_results(prediction, confidence, article_text)
            
            # Save prediction to database
            source_type = "manual"
            source_url = None
            if input_method == "Extract from URL":
                source_type = "url"
                source_url = url if 'url' in locals() else None
            elif input_method == "Select Sample Article":
                source_type = "sample"
            
            # save_prediction(article_text, source_type, prediction, confidence, source_url)
            
        except Exception as e:
            st.error(f"❌ Error during prediction: {str(e)}")


if __name__ == "__main__":
    main()