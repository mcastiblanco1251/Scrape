from urllib.request import Request, urlopen
from bs4 import BeautifulSoup
import requests
from googlesearch import search
import time
import wget
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.layout import LAParams
from pdfminer.converter import TextConverter
from io import StringIO
from pdfminer.pdfpage import PDFPage
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from nltk.probability import FreqDist
import fitz
import pdfplumber
import re
import seaborn as sns
import requests
import urllib
import pandas as pd
from requests_html import HTML
from requests_html import HTMLSession
import os
import glob
from selenium import webdriver
import time
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import word_tokenize
from tabula import read_pdf
import streamlit as st
from PIL import Image
import webbrowser
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import numpy as np

row1_1, row1_2 = st.columns((2,3))

with row1_1:
    image = Image.open('C:/Users/Mcastiblanco/Documents/AGPC/DataScience2020/Streamlit/Scrape/bus.jpg')
    st.image(image, use_column_width=True)
    st.markdown('Web App by [Manuel Castiblanco](https://github.com/mcastiblanco1251)')
with row1_2:
    st.write("""
    # Scraping App
    Esta aplicación guarda tu busqueda y hace una agruapación de la información requerida!
    """)
    with st.expander("Contact us 👉"):
        with st.form(key='contact', clear_on_submit=True):
            name=st.text_input('Name')
            mail = st.text_input('Email')
            q=st.text_area("Query")

            submit_button = st.form_submit_button(label='Send')
            if submit_button:
                subject = 'Consulta'
                to = 'macs1251@hotmail.com'
                sender = 'macs1251@hotmail.com'
                smtpserver = smtplib.SMTP("smtp-mail.outlook.com",587)
                user = 'macs1251@hotmail.com'
                password = '1251macs'
                smtpserver.ehlo()
                smtpserver.starttls()
                smtpserver.ehlo()
                smtpserver.login(user, password)
                header = 'To:' + to + '\n' + 'From: ' + sender + '\n' + 'Subject:' + subject + '\n'
                message = header + '\n'+name + '\n'+mail+'\n'+ q
                smtpserver.sendmail(sender, to, message)
                smtpserver.close()

st.header('Application')
st.write('_______________________________________________________________________________________________________')
app_des=st.expander('Description App')
with app_des:
    st.markdown("""
    Este es una aplicación para agrupar la información y realizar un análsis detallado de la información que buscas.
        """)

def get_source(url):
    """Return the source code for the provided URL.

    Args:
        url (string): URL of the page to scrape.

    Returns:
        response (object): HTTP response object from requests_html.
    """

    try:
        session = HTMLSession()
        response = session.get(url)
        return response

    except requests.exceptions.RequestException as e:
        print(e)

def get_results(query):
    if query[0:6]=='search':
#        query = urllib.parse.quote_plus(query)
        response = get_source("https://www.google.com/" + query)
    else:
        query = urllib.parse.quote_plus(query)
        response = get_source("https://www.google.com/search?q=" + query)

    return response
def parse_results(response):

    css_identifier_result = ".tF2Cxc"
    css_identifier_title = "h3"
    css_identifier_link = ".yuRUbf a"
    css_identifier_text = ".IsZvec"

    results = response.html.find(css_identifier_result)

    output = []

    for result in results:

        item = {
            'title': result.find(css_identifier_title, first=True).text,
            'link': result.find(css_identifier_link, first=True).attrs['href'],
            'text': result.find(css_identifier_text, first=True).text

        }

        output.append(item)
    return output

def next_v(query_next):

    if query_next[0:6]=='search':
        response = get_source("https://www.google.com/" + query_next)
        results = response.html.find(".d6cvqb")
        for result in results:
            a=result.find("a", first=True)
        a=str(a)
        c=(a.split('/')[1]).split("'")[0]
    else:
        response = get_source("https://www.google.com/search?q=" + query_next)
        results = response.html.find(".d6cvqb")
        for result in results:
            a=result.find("a", first=True)
        a=str(a)
        c=(a.split('/')[1]).split("'")[0]
    return c

def google_search(query):
    response = get_results(query)
    results=parse_results(response)
    return results

def get_pdf_file_content(path_to_pdf):

    resource_manager = PDFResourceManager(caching=True)
    out_text = StringIO()

    codec = 'utf-8'
    laParams = LAParams()

    text_converter = TextConverter(resource_manager, out_text, laparams=laParams)
    fp = open(path_to_pdf, 'rb')

    interpreter = PDFPageInterpreter(resource_manager, text_converter)
    for page in PDFPage.get_pages(fp, pagenos=set(), maxpages=0, password="", caching=True, check_extractable=True):
        interpreter.process_page(page)

    text = out_text.getvalue()
    fp.close()
    text_converter.close()
    out_text.close()

    return text

def bag_of_words(ListWords):
    all_words = []
    for m in ListWords:
#        for w in m:
         all_words.append(m.lower())
    all_words1 = FreqDist(all_words)
    return all_words1

# #Consolidación
st.header('Url de busqueda')
app_des=st.expander('Descripción')
with app_des:
    st.markdown("""
    En esta parte lo que hace es buscar en **Google** el tema que este interesado y genera un archivo en el cual guarda los link donde se encuentra
    la información. Para esto se debe seguir los siguientes pasos:
    * La Busqueda la tienes que ingresar p.e.**(arandanos+pdf)** o para busquedas especializadas y profundas debería hacerse de la
      la siguiente forma
      p.e.**('search?q=uchuva+pdf&hl=en&sxsrf=AOaemvLXdmoSyTnMfsOsBNu8zqNY77JQMQ:
      1637934989668&filter=0&biw=1280&bih=609&dpr=1.5)**
      para mas información entrar en contacto.
    * Definir la ruta(path) a donde de se va a guardar p.e. **('C:/Users/Mcastiblanco/Documents/
      AGPC/DataScience2020/)**;
    * Definir la Carpeta donde queremos que se guarde la busqueda p.e.**(Arandanos)**
    * Finalmente el nombre del archivo donde se guardara la busqueda p.e.**(Busqueda_Arandanos_Lista)**
        """)
with st.form(key='Busqueda', clear_on_submit=False):
    query=st.text_input('Busqueda')
    dir= st.text_input('Dirección a Guardar')#'C:/Users/Mcastiblanco/Documents/AGPC/DataScience2020/'
    carpeta = st.text_input('Carpeta')
    file=st.text_input("Nombre Archivo")
    submit_button = st.form_submit_button(label='Buscar y Guardar')
    if submit_button:
    #query= 'search?q=sacha+inchi+pdf&sxsrf=ALeKk00n5wX5-YM81Tt5e12L4q3vd0flOg:1620904445610&filter=0&biw=1280&bih=610'
        table_total=pd.DataFrame()
#        st.write(query)
#        query= st.text_input('Busqueda')
        results1 = google_search(query)
        results1=pd.DataFrame(results1)
        try:
            latest_iteration = st.empty()
            bar = st.progress(0)
            for i in range (100):
                latest_iteration.text(f'Progreso de Búsqueda {i*(100//(i+1))}%')
                bar.progress(i *(100//(i+1)))
                time.sleep(0.1)
                query=next_v(query)
                results=google_search(query)
                table=pd.DataFrame(results)
                table_total=table_total.append(table, ignore_index=True)
            #query=next_v(query)
            #query

            #    f=next_v(f)
            #    st.write(i)
            #    st.write(query)
            table_general=results1.append(table_total,ignore_index=True)
            # dirName=dir+carpeta
            # try:
            #     os.makedirs(dirName)
            #     st.write(f" El Directorio {dirName} con el archivo {file} fue creado ")
            # except FileExistsError:
            #     st.write(f" El Directorio {dirName} con el archivo {file} ya existe")
            # table_general.to_csv(f'{dirName}/{file}.csv', sep=',', encoding='utf-8', index = False)
        except:
            table_general=results1.append(table_total,ignore_index=True)

        #    print(results)
#        table_general=results1.append(table_total,ignore_index=True)
            dirName=dir+carpeta
            try:
                os.makedirs(dirName)
                st.write(f" El Directorio {dirName} con el archivo {file} fue creado ")
            except FileExistsError:
                st.write(f" El Directorio {dirName} con el archivo {file} ya existe")
            table_general.to_csv(f'{dirName}/{file}.csv', sep=',', encoding='utf-8', index = False)

            st.header('Tabla resumen')
            table_general


            #Filtrado y descargue PDF

st.header('Descargar Archivos')
app_des=st.expander('Descripción')
with app_des:
    st.markdown("""
    En esta parte lo que hace es de un archivo donde presente urls p.e.**(https://scielo.conicyt.cl/pdf/rchnut/v39n1/art05.pdf)** con extensión **pdf**
    los descarga. Para esto se deben seguir los siguientes pasos:
    * Definir la ruta(path) a donde de se va a guardar p.e. **('C:/Users/Mcastiblanco/Documents/
      AGPC/DataScience2020/)**;
    * Definir la Carpeta donde queremos que se guarde la busqueda p.e.**(Arandanos)**
    * Finalmente el nombre del archivo donde se encuentra las **Urls** de la extensión **pdf** p.e.**(Busqueda_Arandanos_Lista)**
        """)
with st.form(key='Descargar', clear_on_submit=False):
    dir= st.text_input('Dirección a Guardar')
    c=st.text_input('Carpeta')
    file=st.text_input('Archivo')
    submit_button = st.form_submit_button(label='Descargar')
    if submit_button:
        dirName=dir+c
        dirName2=dirName+'/pdf'
        os.makedirs(dirName2)
        df=pd.read_csv(f'{dirName}/{file}.csv')
        urls=df.link
        latest_iteration = st.empty()
        bar = st.progress(0)
        l=len(urls)
        for ix, url in enumerate(urls):
            #print(url)
            latest_iteration.text(f'Progreso de Descarga {ix*(100//l)}%')
            bar.progress(ix*(100//l))
            time.sleep(0.1)
            try:
                if url[12:24]=='researchgate':
                    url_(url)
                else:
                    path = dirName2
#                    urllib.request.urlretrieve(url, path)
                    wget.download(url,out = path)
                    time.sleep(1)
            except:
                pass



# #st.header('Guardar la busqueda')
# #with st.form(key='Guardar Busqueda', clear_on_submit=False):
# #    submit_button = st.form_submit_button(label='Descargar y Guardar')
# #    if submit_button:
#
#Generar archivo consolidado de pdf
st.header('Consolidado pdf')
app_des=st.expander('Descripción')
with app_des:
    st.markdown("""
    En esta parte lo que hace es de los archivos  de **pdf** guardados en una carpeta, los consolida toda la información de Texto y
    y hace un análisis de palabras para poder agrupar la información si así se requiere generando un archivo donde esta consolidada
    la información p.e. **(table_generalFiltradoArandanos_pdfs.csv)**. Para esto se deben seguir los siguientes pasos:
    * Definir la ruta(path) a donde de se va a guardar p.e. **('C:/Users/Mcastiblanco/Documents/
    AGPC/DataScience2020/)**;
    * Definir la Carpeta donde queremos que se guarde la busqueda p.e.**(Arandanos)**
    * Finalmente el nombre del archivo donde se encuentra las **Urls** de la extensión **pdf** p.e.**(Busqueda_Arandanos_Lista)**
    """)
with st.form(key='Consolidar', clear_on_submit=False):
    dir= st.text_input('Dirección a Guardar')
    c=st.text_input('Carpeta')
    dirName=dir+c
    dirName2=dirName+'/pdf'
    path = dirName2+'/'
    submit_button = st.form_submit_button(label='Consolidar')

    if submit_button:
        extension = 'pdf'
        os.chdir(path)
        files = glob.glob('*.{}'.format(extension))
        l=(len(files))
#        f=st.slider('Numero de Archivos a analizar', 0, len(files), len(files))

    #for carpeta in d:

        latest_iteration = st.empty()
        bar = st.progress(0)
#
        total_general_pdf=pd.DataFrame()
        for ix, file in enumerate(files):
        #for file in files[0:3]:
            #st.write(ix)
            latest_iteration.text(f'Progreso de Descarga {ix*(100//l)}%')
            bar.progress(ix*(100//l))
            time.sleep(0.1)
            path_pdf=path+file
            try:

                pdf=get_pdf_file_content(path_pdf)
                pdf=pdf.lower()
                tokens = word_tokenize(pdf)
                # remove all tokens that are not alphabetic
                words = [word for word in tokens if word.isalpha()]
                stop_words = stopwords.words('spanish')
                words = [w for w in words if not w in stop_words]
                words_top= bag_of_words(words)
                table={'Archivo':file,'Texto':pdf, 'Palabras Comunes':[words_top.most_common(20)]}
                tables=pd.DataFrame(table)
                total_general_pdf=total_general_pdf.append(tables, ignore_index=True)
            except:
                pass
        st.header('Tabla Consolidado Pdf')
        total_general_pdf.iloc[:,0:2]
        total_general_pdf.to_csv(f'{dir}/{c}/table_generalFiltrado{c}_pdfs.csv', sep=',', encoding='utf-8', index = False)

st.header('Modelo de Agrupación de la Información')
app_des=st.expander('Descripción')
with app_des:
    st.markdown("""
    En esta parte lo que hace es agrupar los documentos analizados usando un algoritmo de ML **K-means** de acuerdo a su afinidad
    para aterrizar la búsqueda que se esta haciendo, aplica para un archivo donde se encuentre el texto o análisis de palabras;
    la resultante es que obtiene un archivo donde aparecen los archivos pdfs analizados con un número asociado correspondente al
    mismo grupo de afinidad (clusters) de acuerdo a las necesidades de la búsqueda p.e. si es manufactura, factibilidad, exportación,..
    etc. Para esto se deben seguir los siguientes pasos:
    * Definir la ruta(path) a donde de se va a guardar p.e. **('C:/Users/Mcastiblanco/Documents/
    AGPC/DataScience2020/)**;
    * Definir la Carpeta donde queremos que se guarde la busqueda p.e.**(Arandanos)**
    * Número de Clusters o Agrupaciones que quiere realizar.
    * Número de palabras que quiere agrupar.
    """)
with st.form(key='Agrupación', clear_on_submit=False):
    st.subheader('Datos de Entrada para Agrupación por K-means')
    dir= st.text_input('Dirección a Guardar')
    c=st.text_input('Carpeta donde esta el Archivo a clasificar')
    f=st.text_input('Nombre de archivo a Clasificar')
    #cls=st.text_input('Palabras claves para el modelo de por palabras clave')
    k=st.slider('Numero de Grupos K-means',0,10,4)
    np=st.slider('Numero de palabras clasificadas K-means',30,80,50)

    #model = st.radio("Seleccione el modelo de Agrupación",('Por palabras clave', 'K-means'))

    #st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)

    # submit_button = st.form_submit_button(label='Agrupar')
    # if submit_button:
    #     if model=='Por palabras clave':
    #         dirName='C:/Users/Mcastiblanco/Documents/AGPC/DataScience2020/'+c
    #         df=pd.read_csv(f'{dirName}/{f}.csv')
    #
    #         #df.loc[:,['Archivo','Texto']]
    #         p_comunes=df['Palabras Comunes']
    #         #p_comunes
    #         cls=['comercialización', 'mercado', 'producto', 'producción','exportación', 'proyecto']
    #         cls
    #         lista_total=[]
    #
    #
    #             list1=p#_comunes[i]
    #         for ix, p in enumerate(p_comunes):#range (len(p_comunes)):
    #
    #             try:
    #             lista=[]
    #
    #                     list_=list1[j][0]
    #                 for j in range (20):
    #                     st.write(list_)
    #                     lista.append(list_)
    #             except:
    #                 pass
    #             a=set(lista)
    #             intersection = a.intersection(cls)
    #             intersection_as_list = list(intersection)
    #             #print(intersection_as_list)
    #             if len(intersection_as_list)>=1:
    #                 lista_=1
    #             else:
    #                 lista_=0
    #             lista_total.append(lista_)
    #             df['Clasificacion']=lista_total
    #             lista_total

    #         df.loc[:,['Archivo', 'Clasificacion']]
    submit_button = st.form_submit_button(label='Agrupar')
    if submit_button:
        dirName=dir+c
        df=pd.read_csv(f'{dirName}/{f}.csv')

        document=[]
        l=len(df)

        for i in range (l):

            document.append(df['Texto'][i])
        #document
        vectorizer = TfidfVectorizer(stop_words=stopwords.words("spanish"))
        X = vectorizer.fit_transform(document)
        true_k =k
        model = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1)
        model.fit(X)
        order_centroids = model.cluster_centers_.argsort()[:, ::-1]
        terms = vectorizer.get_feature_names()
        p=[]
        for i in range(true_k):
            #st.write('Cluster %d:' % i)
            for ind in order_centroids[i, :np]:
                 #st.write(' %s' % terms[ind])
                  p.append(terms[ind])
        #p
        a=0
        div=len(p)//true_k
        #for i in range(4):
        #k=4
        f={}
        for j in range (k):
            f[j]={'Cluster':j,'Palabras':p[j*div:j*div+div]}
        f=pd.DataFrame(f)
        x=f.T
        st.subheader('Tabla Clasificada con Palabras')
        x
        table={}
        for i in range (len(df)):

            X = vectorizer.transform([df.Texto[i]])
            predicted = model.predict(X)
            table[i]={'Archivo':df.Archivo[i], 'Cluster':predicted}

        clas=pd.DataFrame(table)
        a=clas.T
        st.subheader('Tabla Archivo con Cluster')
        a
        a.to_csv(f'{dir}/{c}/Archivo_clusters{c}.csv', sep=',', encoding='utf-8', index = False)

st.header('Selección de los Archivos a Analizar')
app_des=st.expander('Descripción')
with app_des:
    st.markdown("""
    En esta parte lo que hace es a partir de un archivo donde se encuentran ya los archivos asociados a un grupo, se
    indica el grupo o cluster de interés para generar un archivo donde se encuentran únicamente los archivos agrupados de
    pdfs. Para esto se deben seguir los siguientes pasos:
    * Definir la ruta(path) a donde de se va a guardar p.e. **('C:/Users/Mcastiblanco/Documents/
    AGPC/DataScience2020/)**;
    * Definir la Carpeta donde queremos que se guarde la busqueda p.e.**(Arandanos)**
    * Nombre de archivo donde están los archivos con el grupo quiere realizar.
    * Seleccionar el cluster o grupo de interes de acuerdo .
    """)
with st.form(key='Seleccion', clear_on_submit=False):
    dir= st.text_input('Dirección a Guardar')
    car=st.text_input('Carpeta donde esta el archivo a Seleccionar')
    f=st.text_input('Nombre de archivo a Seleccionar')
    k=st.number_input('Número Total de Clusters o Grupos del archivo',3)
    c=st.slider('Seleccione Cluster que aplique de acuerdo a su análisis o sugerido',0,k-1,2)
    dirName=dir+car

    submit_button = st.form_submit_button(label='Selección Cluster')

    if submit_button:
        a=pd.read_csv(f'{dirName}/{f}.csv')
        #a
        #c
        a[f'Cluster{c}']=a.Cluster.str[1]
        text_sel=a.loc[a[f'Cluster{c}']==f'{c}',['Archivo',f'Cluster{c}']]
        st.subheader('Tabla Arichivos Cluster Seleccionado')
        text_sel
        st.markdown(f'Archivos seleccionados {len(text_sel)} de un total de {len(a)}')
        text_sel.to_csv(f'{dir}/{car}/Archivo_clusters_seleccion{car}.csv', sep=',', encoding='utf-8', index = False)

st.header('Extracción de Gráficas y Tablar de Pdfs seleccionados')
app_des=st.expander('Descripción')
with app_des:
    st.markdown("""
    En esta parte lo que hace de acuerdo a un archivo donde estén archivos pdfs extrae las gráficas y tablas que estén en
    los archivos pdfs. Para esto se deben seguir los siguientes pasos:
    * Definir la ruta(path) a donde de se va a guardar p.e. **('C:/Users/Mcastiblanco/Documents/
    AGPC/DataScience2020/)**;
    * Definir la Carpeta donde queremos que se guarde la busqueda p.e.**(Arandanos)**
    * Nombre del archivo donde estén los archivos a descargar.
    * Nombre de la carpeta donde se quiere guardar la información.
    """)

with st.form(key='Des_clus', clear_on_submit=False):
    dir= st.text_input('Dirección a Guardar')
    ca=st.text_input('Carpeta donde esta el archivo a Descargar')
    f=st.text_input('Nombre de archivo a Seleccion de Cluster')
    nca=st.text_input('Nombre carpeta a Descargar las fotos y tablas')
    dirName=dir+ca
    submit_button = st.form_submit_button(label='Descargar Selección')
    if submit_button:
        a=pd.read_csv(f'{dirName}/{f}.csv')
        files=list(a.Archivo)
        latest_iteration = st.empty()
        bar = st.progress(0)
        l=len(files)
        for ix, file in enumerate(files):
            latest_iteration.text(f'Progreso de Descarga {(ix+1)*(100//l)}%')
            bar.progress((ix+1)*(100//l))
            time.sleep(0.1)

            path=f'{dir}/{ca}/{nca}/{file}'
            os.makedirs(path)
            #print(file)
            try:
                pdf=fitz.open(f'{dir}/{ca}/pdf/{file}')

                for pn,page in enumerate(pdf.pages(), start=1):
                    for imn, img in enumerate(page.getImageList(), start=1):
                        xref=img[0]
                        pix=fitz.Pixmap(pdf,xref)
                        if pix.n>4:
                            pix=fitz.Pixmap(fitzcs.RGB, pix)
                        pix.writePNG(path+f'/{file}_image_page{pn}_{imn}.png')
            except:
                pass
            try:
                df = read_pdf(f'{dir}/{ca}/Pdf/'+f'{file}', pages='all', multiple_tables=True)
                for i in range (len(df)):
                    df[i].to_csv(path+f'/tabla{file}_{i}.csv', sep=',', encoding='utf-8', index = False)
            except:
                pass
