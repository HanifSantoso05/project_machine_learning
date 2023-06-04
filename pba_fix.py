from streamlit_option_menu import option_menu
import streamlit as st
import pandas as pd 
import numpy as np
import regex as re
import jsons as json
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
# from sklearn.naive_bayes import MultinomialNB
import pickle5 as pickle 
from sklearn.metrics import confusion_matrix, accuracy_score
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Sentimen Analysis",
    page_icon='https://cdn-icons-png.flaticon.com/512/1998/1998676.png',
    layout='centered',
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.extremelycoolapp.com/help',
        'Report a bug': "https://www.extremelycoolapp.com/bug",
        'About': "# This is a header. This is an *extremely* cool app!"
    }
)
st.write("""
<center><h2 style = "text-align: justify;">Aplikasi Analisis Sentimen Pendapat orang tua terhadap pembelajaran daring pada masa Covid-19 dengan algoritma naive bayes</h2></center>
""",unsafe_allow_html=True)

with st.container():
    with st.sidebar:
        selected = option_menu(
        st.write("""<h3 style = "text-align: center;"><img src="https://cdn-icons-png.flaticon.com/512/1998/1998676.png" width="120" height="120"></h3>""",unsafe_allow_html=True), 
        ["Home","Dataset", "Implementation", "Tentang Kami"], 
            icons=['house', 'bar-chart', 'check2-square', 'person'], menu_icon="cast", default_index=0,
            styles={
                "container": {"padding": "0!important", "background-color": "#412a7a"},
                "icon": {"color": "white", "font-size": "18px"}, 
                "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "color":"white"},
                "nav-link-selected":{"background-color": "#412a7a"}
            }
        )
        # st.write("""
        # <div style = "position: fixed; left:50px; bottom: 10px;">
        #     <center><a href="https://github.com/HanifSantoso05/Aplikasi-Web-Klasifikasi-Penyakit-Anemia"><span><img src="https://cdns.iconmonstr.com/wp-content/releases/preview/2012/240/iconmonstr-github-1.png" width="40px" height="40px"></span></a><a style = "margin-left: 20px;" href="http://hanifsantoso05.github.io/datamining/intro.html"><span><img src="https://friconix.com/png/fi-stluxx-jupyter-notebook.png" width="40px" height="40px"></span></a> <a style = "margin-left: 20px;" href="mailto: hanifsans05@gmail.com"><span><img src="https://cdn-icons-png.flaticon.com/512/60/60543.png" width="40px" height="40px"></span></a></center>
        # </div> 
        # """,unsafe_allow_html=True)

    if selected == "Home":
        st.write("""<h3 style = "text-align: center;">
        <img src="https://onlinelearning.binus.ac.id/files/2021/10/Web-Banner-Artikel-BOL-31-06.jpg" width="650" height="300">
        </h3>""",unsafe_allow_html=True)
        # st.write("""
        # Anemia adalah suatu kondisi di mana Anda kekurangan sel darah merah yang sehat untuk membawa oksigen yang cukup ke jaringan tubuh Anda. Penderita anemia, juga disebut hemoglobin rendah, bisa membuat Anda merasa lelah dan lemah.
        # """)

    elif selected == "Dataset":
        st.write("#### Deskripsi Dataset")
        st.write(""" <p style = "text-align: justify;">Dataset ini diambil dari hasil crawling data berita yang diambli melalui beberapa paltform berita online. Selanjutnya data berita tersebut akan diklasifikasikan ke dalam dua kategori sentimen yaitu negatif dan positif kemudian dilakukan penerapan algoritma Multinomial Naive Bayes untuk mengetahui nilai akurasinya.</p>""",unsafe_allow_html=True)
        st.write("#### Preprocessing Dataset")
        st.write(""" <p style = "text-align: justify;">Preprocessing data merupakan proses dalam mengganti teks tidak teratur supaya teratur yang nantinya dapat membantu pada proses pengolahan data.</p>""",unsafe_allow_html=True)
        st.write(""" 
        <ol>
            <li>Case folding merupakan tahap untuk mengganti keseluruhan kata kapital pada dataset agar berubah menjadi tidak kapital.</li>
            <li>Cleansing yaitu merupakan proses untuk menghilangkan semua simbol, angka, ataupun emoticon yang terdapat didalam dataset</li>
            <li>Slangword Removing yaitu satu proses yang dilakukan untuk mendeteksi dan menghilangkan kata-kata yang tidak baku di dalam dataset</li>
            <li>Steaming yaitu proses yang digunakan untuk menghilangkan semua kata imbuhan dan merubahnya menjadi kata dasar.</li>
        </ol> 
        """,unsafe_allow_html=True)
        st.write("#### Dataset")
        df = pd.read_csv(r"C:\Users\HP\DataBerita.csv")
        st.write(df)
    elif selected == "Implementation":
        #Getting input from user
        word = st.text_area('Masukkan kata yang akan di analisa :')

        submit = st.button("submit")

        if submit:
            def prep_input_data(word, slang_dict):
                #Lowercase data
                lower_case_isi = word.lower()

                #Cleansing dataset
                clean_symbols = re.sub("[^a-zA-Zï ]+"," ", lower_case_isi)

                #Slang word removing
                def replace_slang_words(text):
                    words = nltk.word_tokenize(text.lower())
                    words_filtered = [word for word in words if word not in stopwords.words('indonesian')]
                    for i in range(len(words_filtered)):
                        if words_filtered[i] in slang_dict:
                            words_filtered[i] = slang_dict[words_filtered[i]]
                    return ' '.join(words_filtered)
                slang = replace_slang_words(clean_symbols)

                #Steaming Data
                factory = StemmerFactory()
                stemmer = factory.create_stemmer()
                stem = stemmer.stem(slang)
                return lower_case_isi,clean_symbols,slang,stem
            
            #Kamus
            with open('combined_slang_words.txt') as f:
                data = f.read()
            slang_dict = json.loads(data)

            #Dataset
            Data_ulasan = pd.read_csv(r"C:\Users\HP\datapba_prep.csv")
            ulasan_dataset = Data_ulasan['Steaming']
            sentimen = Data_ulasan['Label']

            # TfidfVectorizer 
            tfidfvectorizer = TfidfVectorizer(analyzer='word')
            tfidf_wm = tfidfvectorizer.fit_transform(ulasan_dataset)
            tfidf_tokens = tfidfvectorizer.get_feature_names_out()
            df_tfidfvect = pd.DataFrame(data = tfidf_wm.toarray(),columns = tfidf_tokens)
            with open('model.pkl', 'rb') as file:
                loaded_model = pickle.load(file)
            
            # with open('tfidf.pkl', 'rb') as file:
            #     loaded_data_tfid = pickle.load(file)
            
            # tfidf_wm = loaded_data_tfid.fit_transform(names)

            #Train test split
            training, test = train_test_split(tfidf_wm,test_size=0.2, random_state=1)#Nilai X training dan Nilai X testing
            training_label, test_label = train_test_split(sentimen, test_size=0.2, random_state=1)#Nilai Y training dan Nilai Y testing    

            # model
            clf = loaded_model.fit(training,training_label)
            y_pred=clf.predict(test)

            #Evaluasi
            akurasi = accuracy_score(test_label, y_pred)

            # Inputan 
            lower_case_isi,clean_symbols,slang,stem = prep_input_data(word, slang_dict)
            
            # #Prediksi
            v_data = tfidfvectorizer.transform([stem]).toarray()
            y_preds = clf.predict(v_data)

            st.subheader('Preprocessing')
            st.write(pd.DataFrame([lower_case_isi],columns=["Case Folding"]))
            st.write(pd.DataFrame([clean_symbols],columns=["Cleansing"]))
            st.write(pd.DataFrame([slang],columns=["Slang Word Removing"]))
            st.write(pd.DataFrame([stem],columns=["Steaming"]))

            st.subheader('Akurasi')
            st.info(akurasi)

            st.subheader('Prediksi')
            if y_preds == "positif":
                st.success('Positive')
            else:
                st.error('Negative')

    elif selected == "Tentang Kami":
        st.write("##### Mata Kuliah = Pembelajaran Bahasa Alami - A") 
        st.write('##### Kelompok 1')
        st.write("1. Muhammad Hanif Santoso (200411100074)")
        st.write("2. Alfito Wahyu Kamaly (200411100079)")
        st.write("3. Fajrul Ihsan Kamil (200411100172)")
        