# streamlit_py
import os, re
from io import BytesIO
import numpy as np
import streamlit as st
from PIL import Image, ImageOps
from fastai.vision.all import *
import gdown

# ======================
# 페이지/스타일
# ======================
st.set_page_config(page_title="Fastai 이미지 분류기", page_icon="🤖", layout="wide")
st.markdown("""
<style>
h1 { color:#1E88E5; text-align:center; font-weight:800; letter-spacing:-0.5px; }
.prediction-box { background:#E3F2FD; border:2px solid #1E88E5; border-radius:12px; padding:22px; text-align:center; margin:16px 0; box-shadow:0 4px 10px rgba(0,0,0,.06);}
.prediction-box h2 { color:#0D47A1; margin:0; font-size:2.0rem; }
.prob-card { background:#fff; border-radius:10px; padding:12px 14px; margin:10px 0; box-shadow:0 2px 6px rgba(0,0,0,.06); }
.prob-bar-bg { background:#ECEFF1; border-radius:6px; width:100%; height:22px; overflow:hidden; }
.prob-bar-fg { background:#4CAF50; height:100%; border-radius:6px; transition:width .5s; }
.prob-bar-fg.highlight { background:#FF6F00; }
.info-grid { display:grid; grid-template-columns:repeat(12,1fr); gap:14px; }
.card { border:1px solid #e3e6ea; border-radius:12px; padding:14px; background:#fff; box-shadow:0 2px 6px rgba(0,0,0,.05); }
.card h4 { margin:0 0 10px; font-size:1.05rem; color:#0D47A1; }
.thumb { width:100%; height:auto; border-radius:10px; display:block; }
.thumb-wrap { position:relative; display:block; }
.play { position:absolute; top:50%; left:50%; transform:translate(-50%,-50%); width:60px; height:60px; border-radius:50%; background:rgba(0,0,0,.55); }
.play:after{ content:''; border-style:solid; border-width:12px 0 12px 20px; border-color:transparent transparent transparent #fff; position:absolute; top:50%; left:50%; transform:translate(-40%,-50%); }
.helper { color:#607D8B; font-size:.9rem; }
.stFileUploader, .stCameraInput { border:2px dashed #1E88E5; border-radius:12px; padding:16px; background:#f5fafe; }
</style>
""", unsafe_allow_html=True)

st.title("이미지 분류기 (Fastai) — 확률 막대 + 라벨별 고정 콘텐츠")

# ======================
# 세션 상태
# ======================
if "img_bytes" not in st.session_state:
    st.session_state.img_bytes = None
if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None

# ======================
# 모델 로드
# ======================
FILE_ID = st.secrets.get("GDRIVE_FILE_ID", "14jGzKTiAdbBHoZkgAjl6RO3npVLFqH6k")
MODEL_PATH = st.secrets.get("MODEL_PATH", "model.pkl")

@st.cache_resource
def load_model_from_drive(file_id: str, output_path: str):
    if not os.path.exists(output_path):
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, output_path, quiet=False)
    return load_learner(output_path, cpu=True)

with st.spinner("🤖 모델 로드 중..."):
    learner = load_model_from_drive(FILE_ID, MODEL_PATH)
st.success("✅ 모델 로드 완료")

labels = [str(x) for x in learner.dls.vocab]
st.write(f"**분류 가능한 항목:** `{', '.join(labels)}`")
st.markdown("---")

# ======================
# 라벨 이름 매핑: 여기를 채우세요!
# 각 라벨당 최대 3개씩 표시됩니다.
# ======================
CONTENT_BY_LABEL: dict[str, dict[str, list[str]]] = {

     labels[0]: {
       "texts": ["소원의 별의 힘으로 포켓몬이 강해진다", "스킬이 다이맥스 전용으로 변경된다", "포켓몬의 크기가 거대해진다"],
       "images": ["https://lh3.googleusercontent.com/bLSWXW-3nZoueDIo7-3Eh8NvlvfOT951i_UUVobwZMty2t2MScUUuYyW-KxsUL9O2udYnnl_DqMwbJfPTmenraEpe1by6-q-OSRURAOlL1r_Nw=e365-w1128", "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSwthIjI9h2DzRSqCZs6GJb2YgvDxdmaN6-6w&s"],
       "videos": ["https://www.youtube.com/watch?v=bHMxGDIVBxM"]
     },
labels[1]: {
       "texts": ["키스톤의 힘으로 포켓몬이 진화한다", "메가진화의 영향으로 외형이 변화한다", "포켓몬의 능력치가 상승한다"],
       "images": ["https://i.namu.wiki/i/515KNWadwMqG2a0lOynZiaBRJ4kuY2hqTqMoLC1ak1EKiGIvOeNNwUsWBsZo2UYffgMGCXrH6B9JYV2Pt91m9Q.webp", "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/2wCEAAkGBxMTEBISEhEVFRIVGBUVFRYVFRUYEREVGBcXFhYWFhcYHSggGBslGxkXITIhJSkrLi4uFx8zODMtNygwLisBCgoKDg0OGhAQFy8lHR8rKysrLS0tLS03LS0tLS8rKy0tKy0rKy0rLS0tKystLS0rLS0tLS0tLS0tLSsrLS0tLf/AABEIAOEA4QMBIgACEQEDEQH/xAAbAAEAAgMBAQAAAAAAAAAAAAAABQYCAwQBB//EAEgQAAICAQIDBAYGBwMKBwAAAAECAAMRBBIFITETQVFhBiIycYGRIzNCUoKhFFNicpKxwUOT0RUkNERjc4OistMHVFWjs8Lx/8QAGQEBAQEBAQEAAAAAAAAAAAAAAAECAwQF/8QAHxEBAQACAgMBAQEAAAAAAAAAAAECEQMhEjFBE1Ei/9oADAMBAAIRAxEAPwD7jERAREQEgeJekGG7PTqLHxksT6g+Xte/IHgSeU0+lvEGGzTpjc+N3PAwc7VOOqkK7EDqKyPtSNLpRWM5OSByA7S1z095OPcAO4CZt0xllrqOfXX3Pk3ahwO9a27Ose9hgn35HxnDXpqG9YIHP3zlyfxtn+c7zog532lS3LCDnVWfLPtN+2R7gJlZUVOPlPPlb/XG2sKWZPYd1/ddgPlnH5Tr4b6VOGKswuVThsALqEwcMcclsAPgF+JnLtnTwTh6W/pVLcjmq9GHtVO6tWSp8+y5jockHqZePK2621hbv2tuj1aWoHrYMp7x1B8CDzB8jzE3ygUat9NY77fXrOy9F6XKOhHnghlJ8dp6nF60mpWxFsQ5RwGU+IIyJ3xy2645bbYiJpoiIgIiICIiAiIgIiICIiAiIgIiICIiAiIgUPi7E65znobR3dyaYL+Tv8zOOol7WY81rOxPDdgGxvfzC5/ZPiZJeluiavULeB6jnnjubaFcH3qEYf7th1IkJw52P0SEdo9upOTzCILWZrD5AMvxYTz5425an1wzl238Q4gKw2Eax1Uv2dYBfaM8zkgKOR6nng4zK7x3jV7fog0967LEZ7jWnOo4UrXvcEnv5gDp3Zn1Dg3DK6UGzDZ5l8glzjG4nvPLrKlxrglaalxSwQMFZu1Qtp62fcQFKENWDjPMFRkcx0m/ymM6ejgx48bvkQfonoG1GuRLbHsQVWs4ZnKgeoqnmeR3Hl8ZfvRrhyUvqmrzsZ1QZZjns1wxG4nA3s4wOXqnxmj0e4C9VZ+krHaYNj1ZLPjotbHlWnX7x5nnnnLDXWFAVQAoAAA5AAdABOmM1NN8txyz3jNRw8Q4LRcxayvLEAFg9iMQOmSjDM80fB0pQJS91aLuOBazgZJY/W7scyTMPSG/VJWDpK67G5795O5Vx1rQEB28iw+PSV/0KLarUXtqXa4IK8LYBsrfc+cVYAVgAOozFs25J79JdvqNRfce7aunNXxsKKMe5iZMcOW0Vjtyhs557MEKB3Dn1x48s+AnTEqkREBERAREQEREBERAREQEREBERAREQERPGYAZJwPE9IGvU6dbEKOoZWGCD0M+Wtpl4dxJrg1jpusQo3rFKmSliqd5zlWwck9lifQ/SXXGnS2Opw3qop5eq1jCsNz8C2fhPnWnR9eGp3/5zX9JUbCSt1YyDWz8zvXecNz5PzzjIzfbX55ZY3KfF59HNFTXUf0Zw2nc761Ug11jABCY6Dl07vfmRHpHUw1DDJUWohRh1V6ywYZ8cMhweoLdRmVChtXpHOwXUMT6w2bq3PmCGRj5jn5zvv4nrNQVpvBTl2lbio1ncpCn1W9oYbyGCfhrx8pqMZTyDrRQdzbqrB/aafdsfzdF/k4I85t1np7Y9dYpKi/LCwKm+txjKurHlWOuVPMefInh/wAnagIlllLbXG4OgLoR3E45ry54I7+pijSO/sAv5IrOT8hgfEicJ5Y9Oc3HWnpTrAo3XISAM4pXBPfgcz17pdPRvh121NTqW26hwd6qqoChz2a2YHrMq469DnHLOeD0c9EdrJdqANykMlec7GHMM5HIsPAZAPeeRFwnTGX63N/SIibaIiICIiAiIgIiICIiAiIgIiICIiAiIgJBenFG/QXDGcdmxBAIwliOeR8gZOzTrNOLK3rPR1Kn4jELLq7fD9P6R6hKV0pYPQXpUB87qQLExsb7vdtOfLEzrdlZXRiroQysOqsOh8/AjvBInBxrRsltlbAq2WBB+ywPPHiM8we8EGdGlv3oG8evkRyYfPM4Xb7XFjh3qdV9R4L6aVXIFKkavp2A62EDJesnka/Enp0PPGdWt4fqrTY21Vd0KhmbFdQ9baqgZYkZySQMk/AVn0R4QXerVNkKl6JVjlubnvfP3cZTHed3gJaqPSTt9dbpKbEU1L7TLvW6wEixBhht2cgfEk9Npz3wy1Nvk83HJnZj6ie4dqAU27djV4VkJ9kAciD3qRzB/kQQOtTkZHMHp4GV7Vobt1bfQako6AElq7UIwQCMdonPPcy56DPP3heodKlcKWTmttSgl6LByfsgPaXdn1OuDkfdhyWGJq02oSxQ6MGU9CDkf/vlNsBERAREQEREBERAREQEREBERAREQEREBExssCjLEAeJIA/Oav0yv9Yn8S/4wN8Tjt4rQvtX1j3uuf5zgu9J6eYrW25h3JWwU/8AEs2p/wA0slo4vTL0SGrXfWQl4GAWzssHcHwCRjuYA4z0M+Yp6N62vUmgUsGsxzxurQk7RbuXI28uYOM7fHr9C4l6S6noq10599tgHl0VW/iE2+grFv0u57Gcmxa9zNnlXWpPkAGduQwOUZcXW678fPnhOmXpTjR8MJp5dh2Aqznm3aooz4lifzlP/wDDPhdj60W4O2pWLMe9nBCjzJyW+HmJd+N6mnUGivBuqFu60VrY6gLXYVyawf7QJ8cTv0Ov0ta7E+iUdzVvUCT1PrqMnzmbO2ceTxxs13UhrNIlq7bF3DqOoKkdCpHNT5jnIJNJfprmsL9rpnA7QkAX1svJbWCjFnq4ViMHCqcHBMsSOCAQQQehHMH4zKVyRd2gBPaVuarDg70wVsH7aH1XBHf1HcRMl1lqcrKt4+/TzH4q29YH93d8J5ZpXq50jdX305AI/wByxIC/unl0wV553aPXV2ZCt6w9pGBWxf3kbBHv6SjyrjFBYJ2qq56I52WH3I2CflO6abqVdSrqGU9QwBU/Azh/yDpe7T1KfFUVT81xAlIkW3CKhzBsTHP1brQB8N2Jz1OAN1WtZgRld4S2o+fqgMf441RORIajjgDhLiiliqo6ODW7McKpB9ZGJ5AEY5gZJ5SZkCIiAiIgIiICIiAiIgIiIHFxbRV21kWYAU7w3L1CufW58umQc8iCQeRlN0mrpuUk6fSoMnYVWvfYndYyFc1E9dmWI7z3C/ETSdHXjHZpjw2rj5Ymsbq7SqkgUeyFHuAx+Uwu1PrKiqXsf2UXmxx1JzyVR3scDp3kCTnEuHaVcD9FqexzhF2KNx7yTjko6k93vwDt4RwaugNsVQ7ndYyqF3HwA7lHQD+uTOn6/wAieKva/gDmtrNReEUDJSnIwO5e19p2J5DaFyTgAzr4D6OlUIvJNZbemnJLIg2ooFpYntSAuQD6q5xg7QZYrdMrFCwzsO5eZxuwQCR34z/WbJi5W+1eAcsdB4d0yxPMzzMg4reFgNvobsX6kAZps8e0qyAT+0MN5zbo+Ikv2VqdndjIGd1dg7zW+Bux3jAIyMjmDOjM06uhLF2vnHUEHDKR0ZWHMEeMiu2c2s0NduO0QMV5q3R0PijDmp8wRObh2sbd2Nx+lAJV+i3oOW4dwYcty9xPLkZJSCMbRXr9XqAR4XV7/gHRlI953TE3awf6vQ3mNQ4z8DTy+ZkrECGUa5jzGmqH71tzfyrE5dP6Lqbnt1AosDgZrWp1qLg57Qq1jLuOcE7eeBz5SxxLscml4ZTX9XTWn7iKv8hOuIkCIiAiIgIiICIiAiIgIiICYXWhVZmOFUFiT0AAyTM5F65+0tWkc1TFlvhn+yTz5gsfDaM+0IDQqWY3uMO4AVT1qr6hfeereeB3TsLTULAWK59YAE+Wc4/kZmRKMszzMwmIcbiueYAJ8gc4/kYG0tMS0YjEDyez3E8gadXp964zhgQyN3o46N5+BHeCR3zdwzVmxMsNtikpYv3XHXHkeRB7wwjM5LH7O5LPsPiuzyb+yf55Q/vr4SCWiIgIiICIiAiIgIiICIiAiIgIiICIiBhbYFUsxwFBJPgAMmRfC3Ip7Wz1Wszc4PIoCMqp81QKp8wZt42C6pSP7VgG8q19ez5gBPxiaeM80Sr9c61nzT2rB8UVh8ZQ4PWezNjD6S4m1/EZACL+FAi/h853YmRnmIGOJxcN5vqH7jaVHurVUI/jDyQCzi4Iv0Cn7xsf+Ox3/rA7J5Nm2MQNIcEle8AEjwBzj+RmW2cucarH36sj31uAf/kE7sQNe2a9TQHRkbOGBGR1HmPMdfhOkLOXh1xKur+2juje7O5D8UZD8YGfCdUbKgWxvUlHx03qcHA7geo8iJ2SA4bqQmrdM4W4ZA/2qDB+JQL/AHcn4s1QiIkCIiAiIgIiICIiAiIgIiICIiBGI+/VWeFKKnlvs9dh8FFf8c13ndq6k6iuuyxvJmK11/l2vymfCx9c337rCfw4rH5IJr0XPU6l8/qaseGxWf8AnaZRJ4iYbolQutCqzfdBPyGZE+h2pazh+ksbkz01s3kzKCR85s9KHK6HVt3im3HvKED8zNXocQdFUPuGyvHhssdR+QB+Ma+qmcxMgJ7mREbxVcNRb9ywBv3LAa2HuyUb8IkjtmnW0LbW9bey6lTjqMjGR4EdZp4TrGelGf6wZSzHTtEJR8eWQSPIwrt2yF4pb2Nwb7N42n/eoCV+JTf/AHYkubZH8b0xupZF9sYavPQWKdy/AkYPkTLPYr+qsK4tHtVsLB48vbHxQuv4pdEcEAg5BAIPiD0lIovDqrjOGAOD1Ge4jx7pYPRa76Dsv1J2D9zrX8lIH4Z05Z9ZxTMRE4tEREBERAREQEREBERAREQERECJ9HbA+lqcHIYFs+O5if6zPhOCdQR33OP4VRf6SF9D9Xs0OnRiTtQLnvOORJ85L8FYFHIPI23n/wBxpqyxElmebphyjd5SKifTF/8AMNR5qB82Uf1nD6E3epenhYHHudF/+ytOv0v56HU+SFv4SG/pID0WuKavGPVtQof3l9dPy7QfETpJvGr8XffG6Y5ic0MyOpHZ6l1+xcO1HlauEs+BXszjxDHvkjOPitLFA6DNlTCxB97AIdPxIWX3kHugdeJ7iY6ewOiuvNWAZT4gjIPym3YYFU4jpuy1BA+ruy6fs2czanuIw497+E28L1Aq1CMeS2YqbwznNRP4iy/8STvE+Hi6ooTg8ird6ODlW+fI+IJHfKqvroysCrDcjr3o45HB9/MHvGCJ2x/1j4s3q7XuJH8D1/bVAt9Yp2WDwcd/uYYYeTCSE4NEREBERAREQEREBERAREQERECk1V7DYg5BbLQB4DeWH5ESX9Gh9C48LbfzO7+s4NfUE1V6/f2XD3MvZkfOon8Qnb6LMN2pTvDq/wAHQAH5q3ynbLvCMz2mMRibgIKzk0i+O6TtNJqa/v02r80IEqqowpV6vWsULYg6byMMF+PT4y+NXkEePKUrhy4qRe9Rs+KEqf5TrxfYzat2kuWytLEOUdQynxBGRNwSQno1qdpsoP2T2lfmjn1h71fPwZZN7pzs1dKy2xkTDnG2RXDonFdj0/ZObav3SfpFH7rnOO4OuPLuLmcvEtMWUMn1tZ3156FgCCp8mBKk92c9026HVLbWtidDnkeqsCQynwIYEEeIgZ85Ccf0BB/SEBOBi5RzLoOjgd7Lz6dQSOZAlgxPZZdXaKjotX2NgtBzWwAtxzynVXHmp/5S3XlLgrAgEHIPMEdCPESocS0p09vL/R7CSp/U2E57M+CsclT3HK96idPCOI9iRW/1JPqt+pY/ZP7BPQ9xOOhGN5zynlEnXSzxETk0REQEREBERAREQEREBERAgPSevDU2d2TUx8N4yufxKF97CcfAnC6xxyzdSOfeTS55fK78pYOK6PtqXr72HI+DA7lPwYA/CVDS6rFmntI2kWitx3obCaWQ+6wjP7s643eFjN9rqTPC0wMTm0y3yoFdt2or+7azDzW0C7/qdh+GW3MrnGxt1SnHKyo8/wBqpgMfKz8p047rJL6cWpsNe25c76suAPtrj10Pky59xCnulq0OrS6pLa2yjgMp78HuI7iOhB5ggiVfUDKMPFWH5Ga/Rfia0rp1dsV6kJt64W9kDdegD/8AVj701yY/UlXSMT3E9AnJQCR1+KLDZ0qtID46JZ7K2HwDclJ8QviSJLMwvrV1ZHUMrAqwPRgeRBkHu0xtkZodW1dn6NaxJxmiwn65BnKE99qAc/EYbxxJb4VjbUrKVYBlYYKsAVYHqCD1Eq3ENGdOcN61DHCu3PZnkK7SeuegY9cgHn7VqLzXcodSrKGVgQwIyGB6gjvmscvGppB8K4qaMV2kmn7Lk86P2X/Y8G+z38uctAlO1/DXo5oGs0/MEc2toGPnYnd94cvaHTzhXE2pVTWe1055hAclR41N3j9g8vAjpNXGZd4pvXtcomjR6tLUD1sGXy7j3gjqD5Gb5yaIiICIiAiIgIiICIiAlP8ASfh+2xsck1AIDD7F4XkfiAGHnWfES4Tl4loVuqat+h6Ee0jDmrqe4g4I90uN1SuXhWtF1KW/eHMeDAlXX4MCPhOrdK36N2NXdfpbOVg+mxjCtnCu6D7rHa3kbGHUGWHMtGeZD+lFf0SWd9Viv+Fs1vnyw+fwiSuZhqKVsRkcZVwVYeIIwZJdUVm0+qfcf5SIpqBr0VZAK9ngg9CvYbSD851dqw077/rEV0c46umVJx5kZ+MxqpxbUP1VJz5Fyir+VbT1b25p7gvFSpXT3MSTkU2N1swCezY/rAAef2gueoMnd0pXEq81k7d2wrYFHVihDYXwYgEAjmCZZatQawC7b6GAZL/BT07XHTu9foc88YyeGeOq3HfmJmEmQSYVwcS4et9ZrfODggjk9bg5V0PcwPMGRPCeLWJaNJrMduMCu0DFeqXntbA9hyBzHTIOPCWfaJGcf4Mmpq2nCuuTW+M7G8x3qcDI78DoQCKO0LMuzMqPBuP3U2GjVKSqDm3Nrqz9kn9dWRnDjmMYIJztt9GoR1DoyujDKspDKw8QRyMWaAVyH4jwAMxspISw5LKfqrj4sB7Lftjn4gya3xviXSKUjMlpA3UakAEg4IsUdCfs3J+Yz9kyc03pEFH+cLs6DtEy1Xvb7SDzPIfenfrtJXcu2xQw6juZT95WHNT5g5kDquFXVHKHtq/A4GoT3H2bR/CRj7WeW945e09LVVYrAMpDKehBBB9xEzlC01i7yanaqwe2q5rsB/2lTDn5Fl90lE41qExyruHfkmpwPEEAqx8sL75Lx34u1piQyekdX20tT3oXX517gB5nE7dLxWiz6u+p/wB11JHvAPKYssV2RAiQIiICIiAiIgRXG9Bu2Xoub6MlMdbEIxZV7mHT9oKZupcOqupyrAMp8VIyD8p3yKUGh2B+oY7lIBPYuc71OOiH2ge4lgeWIHUEmQrnMeMaf9fX8XUfzgcXoPS+r+8T/GUV/wBIuE1dpZlSGvFZVkd0Pab66WD7Th1Ias4IPR/GdbcJa66+6uxV9ZaQGTcGFO4E5DDHrs4+E28fsqsp5Wr2lbpZXsdN+9SDtXrzYZX8UntNUqIqqMKAAB4e/wA43YK4eC6kEYNDDxJsUj4YbPzEm+E6HsaK6c7tihenL3AHu7p2RLcrfaaVXi2rfR2/RBWoauy01MSNhrZN/Ztz2ZDj1SNuR9nJkzpOILYdvNbAATU+Bao8cZ5jn1BI850arQV2MjOuSmdvMjqVJBA6jKqcH7omWr0ldgAsQMAcjI5qe4qeoPmJNq8zGZzNobF+qtJ/Yu9Zfg4w4PmSw8pgdS6j6Shx4tWe0UfAYc/BTA1cX4RXqNhZV7Ssk1uVDbcjBUg+0hHVfceRAI1aLSjJFYGnvUevWoBosHPDheWRnPrDa3cfCdK8VpP9ooPg+UPxD4ImGrvpYAm5FK81cOm5D4jnzHiOhgZNq3T66sr+2mXqPny9ZfiMDxM6aLVddyMrL4qQR8xNvDr2emt2G1mVWIwRgkZPI8x8ecw1HDKXbe1a7/vgYs924YP5yD2eZmk8JH2Lbk91hb8rNwmscJs/87qP4dL/ANiXYa/h9dy7bEDeByVdfNXUhlPuMgeKaJtOgftGsr3IhVlzf67BV27B9IQSCRjOATk452E8Lc/63f8AAacZ+Iqm3TcKqRg+C1gzh7GLuueR2lvZz5YlmVno0qNfEaR7VqKfBzsPyfE6lt09nWyh+4Zatv6y4zW9KnqoPvAM1eTaaVd9BpwAd/ZY7673qHyRwD8YXXVLhRxUjwHbaV2Pl6yFj85ZBoah0qT+Bf8ACblQDoAPdMWqria6w8q77rPMadTny3bVWe2Ua+zktxoU8tzCk2DzFaq6n4v8JZIkED/kfV/+p2/3Gl/7c9k7EBERAREQODW9/wAZD3dZ7EDdwj6wSwREBERAREQEREDk4n9WZReC/wCm/H+giIH0WIiAiIgIiICIiAiIgIiICIiB/9k="],
       "videos": ["https://www.youtube.com/watch?v=gHXfCWGZWNs"] 
    },
labels[3]: {
       "texts": ["테라스탈 에네지에 의해서 진화한다", "테라스탈과 동일 속성의 스킬이 강해진다", "포켓몬이 보석으로 변한다"],
       "images": ["https://image.toast.com/aaaaahb/SERVICE/bulkUpload/20250113/122625/50/184984683/4521329399584.jpg", "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/2wCEAAkGBxMTEhUTExMWFRUXFxgaGBgYFxgfGBoaFRUYHRgXGBsYHSggGBslGx4XIjEiJSsrLi4uGB8zOjMtNygtMCsBCgoKDg0OGxAQGy0lICUtNS8tLTAyLS0vNS0uLS0vLzgtLS0tLS0tLS4tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLf/AABEIAL0BCwMBIgACEQEDEQH/xAAcAAABBQEBAQAAAAAAAAAAAAAFAAMEBgcCAQj/xAA/EAACAQIEAwUGBAUDAwUBAAABAhEAAwQSITEFQVEGImFxgRMykaGxwRRC0fAHI1Jy8WKC4RUzQ3OSorLCJP/EABoBAAIDAQEAAAAAAAAAAAAAAAADAQIEBQb/xAAvEQACAgEEAQIEBQQDAAAAAAABAgARAwQSITFBE1EFIjJhcYGRofAUQrHBI9Hh/9oADAMBAAIRAxEAPwDcaVKkaIRUqbL00+KAqahJNKmrd4HnXeaohOqVc5xSzUQnVKuGuDrTS3qmoSRSrgPXU1EImNIV49eiiE9pV5Ne0QipUqVEIqVKvGaKIT2lUe5jEX3mA9RUR+PYcT/MXTfXbXerBSehKlgPMIvcA1JodiON212DMfAVDxPHMMw/7q/GhrlGJNu4rb/mHLypqYh/dcU+Q/2wjc7SHlZb1Kj71wO0552W9Cp+9D/Y3Dsqnb8x57Gm2sXR/wCOfJx9NKeMeGZ2fUA9Q7a7S2T72ZP7gQPjtRWxiEcSrA+RqlPdy+9bdfMEj4qYpIsENabKTzG085Gx19fKobTqfpMF1TD6xL1SoVwHiRuqwYQ6mD+vwg+tFayMpU0ZtVgwsRUqVKqy0VKlSohFXhr2uXOlEJDxbgbmB16UJvYlrecPbDQuYHWGAI2+NSeJ3RkJaYmNI6eNS+CnNYXUsNQMwgkTzFNHAuVg7h94XlLWwyFdwdVP9p6+FTcNjAVmhXG8RkxNuboRVghMrazoTI06igmN42BccIZBYwPXlTBj3dSpapbLVwuN4JJj0Mfaoz8YAGp1FVnhPadQ2VycpMg/0n7jy/WhnE8aWuvkJaWcqFkyuYwRG9M9GvqiFc3xLjf44pEjePWoLcfARWndmHwCn71npx7nQEjfeZ3HP4/Ou2uFkiYAckQdO8F0E6nQGBuNJpo04EvvM0Wz2jEeUSelTcPx5Sd9KyY3G2DayG97RhHTkRJiddTWr2MDYQwLNvTnlk6f3TS8uFVkhzDVvGK3OvMRjABvUTDuMwAA67D9NKh3LgYydhMiByOg2rOMfMvvhHC4olookKruFxBUloB86nY7jNqyJY6nSBuY5RUMhviAcQlbkDvGTrQ3jHE7aKVDjPpAGp3HLpQPEcVuXhEm2DoIIzHzOymodu1EgDeZ5k9SSd6cumvlply6sD5VhTE8fvGVRFSDEsZ9QB96hnFXGDB7hYNExAiP6Y1HxrzGEBFumYgBoE67AxQHF8dUe6NOpOvjA5H407CqleuZnd8rGrhpMMkDuAkbltfDc/foa5fGWrUd9VJ0gQDqo6Dx28utUbH8ezf1MI5nSesLA/fMUKxXFS3nOp66CBA0H/NP2QGI+TNGfjdld2JOkBVJJmYB0gExz1NRF45g7kArlkxmKagGNSV9dazZ8aTU/E2XtP7NvfKqSDupaGjziJ8zVGIEYMSzSlvB7U4d5NsbNILW2Mj3tRDH4NTPCuMF7ttGBGZ1Ug/3AEU92fwiYa3avYq85Z0OW2dwGEGYJJXLl6belEbvErK3QBbtLqDnCBiYM76HUZSDy13rIxBuhNwG2rMas8UKhQ494EjnIBI+oNR8O0l3WAmYFZMDbvESYjYedO2eGJfD3PZouWTKMROvMEwJE0F4pxlReyhZtLAKndgIlSQdp+lGMhbuKzqX4Et3ZZSzXLk90mFPULz15Ez8qsVVXg/GkY90ZROg6DpVms3ZFIyNua47Gu1ajlKlSqkvFSpUqIRVy21dVy5ohA2KCDMLvuEj4g7+VDXt3zeVluKUkQEcBQvTLNTuL3lg9elZ/wAUvEN3N50850+da8S7opmqG7nFFxI/D3yFvqSLV3SGP9LR1/etU7ihdr1wXO5cBYso0GmpAgELpsdtPKY92+XJEnVpPT82s7DU/Orzwnsw+ItpcxRyAAQf/KyDUByduWp1jlWr5cXJivrlM4ZgrlxvZ2xcck7LOUyNTGw0MSfjWscPvJZtKuUBsqzAEk5RJJ5mZ1r3AJbtKbdlRbQH1Y/1MTqT59KhcRMGeUlfjLKfncH+0VxPi+fIUDqOB/ubNLjXdRkc4XCNcZmsIzt3tRPOCY26ct5onhfYpolhU/ttoPpQJrkFH/paD/a2h+G9WO1tqIj7Un4XqTlw7WPKn9vH/X5Rmpx7H46M9u5OdlT5qv6GpD2LdwRGQ9QI+Y0NM3X2H7/e1Q7rNEqx05eorpATMTJuFwrKTmjoPHxqFjsM0nKNNzH0qTgcaTCtzHzrnHXGzROh2j6UB6cKezKMo23IcZVUkcyf38KrC2Hza5izakn7dKsN605dljRN/Xb7musLZ9qZJGg0A3jqK03Qi8ZCtZHNcfjIVvDuql2UwBOmsegoBieP+0YqrZBB77RuAYAGwE9atKA5tCR0+5PSlxHgtjEAl0739a6OfGRof9wNMGUA8iZhgB5MH9j8QL+FZW1hmUz0Ov3rN+0eFfD33tNrE5SeatsfP7itQ7O8DOGe4BczI8EAiGBHloREaiNtqgdv+AfiLQuIJuW/iVO49N6XjyBMp9jL7aEyUuetcOelWe72OvpaN10yqImfeEn+ncaxv1qLwfgXt7y2l1n3jGy8z5j51tLrRNyIU/hv2b9tc/E3B/LQ9wf1OOfkv1iu7tsXuJOzCV9sAfJWC6DntWlYXDJYtBEGVEXT0G9Ufs32iu5hcCK6Bm/JbzAAS2oGcQCDO1c7eWJaOxgXzCnFcSbzhnVRPuSCIQzkEqdYkz/aPGh+Ox+Et2rjW3u37lkAsjI9oQxyhgbiju5o1E7jaZEniV03VZiAGZWEKo5qQABmEmdNY5a1nXD+JtaBuWGa0IfMFULIQA3Ea3rbV1TvqygBsjKy7wp2K0BGqoezLl2d44byPANsk5MmcNM97oNY1jeNfJ/jODGQFQ0kLoQdT3h3ecfefCpfZniD3eH3PaZVuYbEvbJRQivEawgAHvcugNGuH2FdLaozC8zqTqsgJOs7gxPjrQDYuRtpqEpvBvbqZVGYeH09OfTSr1w3jVwATabx1GmngfOqDj+072myqSAo74ZFEMHiEk97SDyO9T+EdrGuhQ6TJiQw356NpzHMcvOnBMZHMU7ZgeBNJwvHbbRJykmIbTXprRRWnaqHhcRbuAry3ysNR6c/T40Q4djHsGC2a1OoJkr4g8189aq+nHawx6mzTCW2lXNq4GAI1Brqss2RGh/EL4A3g1NutAqr8dxyweRq6LZlWNCA+O8SOs6+Iqp37uY6kR4/eNRyqVxO+SdDU/snwz22IDsBkt989Dr3F9THoDXSUBFuZ/qMMdnuzqWVF68oz7onTmGcHn0XluddrFaxpLd8+XQeleYm1nMkmfCh2KTIw1Mc+uh1rITvPMbe3qGhZ0zDyYdCOY8DQ7iC91155ZHiU74+Qcf7qm4DEjToRB9P2PnTuLw/5gM0ajxjWPXb1rLqE342Q+0Yh2sGEqOIv5QCADrsdiKtmBIuJodtAecEArPjlImqk1jMoK7AlfgdPlFG+AXIGSf2JI//AH8BXnPhOb085U+Z0tWm5L9oav2O6DzX9Nf34UKvNEgeH2p7E8bRBlHfPQfrt8JqNZu+15FdPeOo9f1r1Sgjucph7Ri5cyrm8Qq+Z3+U/Kpf4spkYiSBJ06n/NMX8G5uIrCFTWeTE9P3sacu6uSBOoAHloK5q3m1xPhBx+n/ALHNSYR7md8exswiwC+/kOvzpux3VCqO8dfPpHkIrx8J3mY77axuD8hNPWjsI7x95uYHMCuqqhVqYwdzWYxeGuUf7jGnl5VJw7gDQg7QBy/flXnss0qDAnfmR6UrthVGmpjnt6VaweJDXdjqOkwwYnWpMhQWJ0jT15UMxGL21M1KtWy6S2ig/m5iNz86qy9XJUGQuJ3ScLfAVnJUgAbksJBHlofQVB7I8CGGtSw/mvq56dF9PrNFfbAmFMjry9BUhahmNbZQjmzB3aXEezwl9+lt488pA+dY/wABxJV0ynvZohoCxpAJJ2OoIMCOZnTSv4k4jLgLg/qKL8WBPyBrG7V2DTcCWpML5msYbG/iSzsAHYswOmXuiWUg/wC2PhUXG8AtOVzopLzOXu5hdUBixWM7FJXMdQGIkTQTsxi8hm4TlAZiBE6KQCJ0zagUT4V2nsohL5ixMCHAcAEMCJQgbASCPKqOnio1R5uHcNYQYdVBAVrjllUd5nPea4zQc0TJA5QOVEsRa9hhXxYtMLotQEA93kbgG40112oXwrtLZdHXD2ZuKrOntWzSyCTAAGuTNEdBVEPbvErivbs2Y7Mp90rzSNgP80sqeo5QBzE3aNLspfDOFAhmClxsGliQYzHQa6HwqZa7Pkj2uFcOAdufx2mhvbPglp7Yx+C1sOe+nOy53Q+HT/FCOzXaN8My96VzaodoMa+uo25U/HkFUwmfNja7UzS+F4xC3s7ujCACRB8PpynzqwiwwH9Q68/Wh1p7WKtKxOaQNY7yyOR6wae4PjXsuLN0hgdEf+qOR6GpZSnK9e0UrrkNP3LL2fchSh2G36UYqJg7Y3FS6xObNzagoVIuPYgaHWqD2gxja5k+FXvifums84zJaNpManT1rRpxzKZJX7dv2jBVBzMdBG/hWg8Pwf4S2Lbe80MxHXUAeIEfM1RMJeKuGG4YEeYrQLzC8iODy89+TfsUj4jqjhdN30fb3/nI/P7S+LHuBA7kxEkA8j0pjG4YsIjUbH7VFwuMNoxGnMHbzU/Y1YMPcVwCOe1Rjzo43IbEhsZHB4MDYFoOU6eH1+X0ovbucvhUbGYTKc41HOlcBy6bj7U1qPMqljgyp4y6bT3LcaFpH78o+FEMDhyUN0GNDHlHePnyHmaZ47aDOrHSTlPgf2SfSpeGb/8AnFsEAgQx/wBxIB6AiNfSvN6bSBfiJRuhyP8AX6X+06WTMf6YEfgZ7bwtpRnknwjX/kUQwjBxCAieulRVClVVhoy/ArA3+MeYrnCK1kXCNQFlT0lgK9MeROdZuGRZIBQ+8BmTx6j40OwK94nWAZ+G0+sU9wnGs+XPrHOoyYjKpY8z9NfrFUx4wGY+TUrkbgSbcYA6/H9aRVWEDT/P6/Shv4suO4rP/qA7vxOnzqPiMcyIzkAAZQddZJCqogQSTAGvOg5MYIG7mQMbnmoRu3Vt+J/fxqLZtXLzSBlXqdvTrTdnFJOZwXfkNMojbzp2/i7rrmkImvdGjeGvjT6IlRUlZbNjU9+58/8Aih+Pxbsj3HMKqlo5aCab4YQWIf3twev+K97Wo34O8EGuT5fm+U1Tpq8yrNxAfZLtfbuRavQlzYN+Vv0NXYV8/um/7/fKti7E483sJbJ95Rkb/bsfUQadqcIT5hKCwagL+LFw+xs2xu1wmP7VI+pFULBcNA7zfUfffSa1Ht9gg9lbkSbZ+TaH5xWY8WxYAyg+G0HUbRMU7TEenK+TIPEsfJyqdOevT7VEsXtQDJE8t/H5UzcUzt8enXSZ/wCancDTvseYXT1Ov0HxqMjAWYxLbiFOEcYGGdbiKZUgyT47QPhS7WNh2uNktqAwDIU0IzgGH5GDoRE6HWlirCNuO9yI+rDY/XxpnAcDL6sxyciN200IkaDx+HWqh0fqT6bjj94uz+PNjOloG4Lgi4jnuMvTKOc8zMH4F84W1ZS5cIUqJgFV1zDuptrJ+AE8qiXcH7AhmcBQwUEZtTBJAEGDAOkxuJoXxPiLX3CqCFB7i89eZjdjp9BVCyqOBzL81RhzsT2hNhwrSUOhE6biCAeY1+Nareti4sHUESCOUHQjy1+NYHhyVaDuK1zsHxP2lnIx1Tb+07fcfCpxtYqZdQlHcJeey3EC6lG95DGvTkfhVhqkcObJikPJ5U+Y1B+tXYVkzptbia9Pk3pcH8YC5dWKms541Mnvg1onGsOWUnT1rNuL2BJ0FO00nJ3I2BRGMNIPVdR6jerLw7BXF71tw6j3su/kVOtVbBpDSDHkat/BkC2ZIIHtBJG/u/anZ1DLRFg+ICxO/aSYaQeh3g+e/lUrCK9o92Cp3U7eh5fSp/FPZNlGjmOfvRyIO81Dwr5epHQ7jyNcB/huz/k05IPt/Ox+M1LqL+V+YTw2PzabHof39K5xVyJPXehHEMKxGa2c0a6aMviR9x8qbwWNe4rLcbQRrEMQZ5jT5UhPioQ7MykH7eYw6bcLQ8Rm5aa6WI90deo5Dx/Wm8Cn8wSSs6T0McxzBH2qa1sQQkidhz0+lN4Dht15lco5E6GQdxzrmZs+XU6gZVXkdV9prRFTHtJj66NkAzMDEAiOu50g78ulLEZxpcVhPkd/7RpXuOs3E7wGcDcr913HpPpXFvBXb7K2qr1afkNz9K6J+JakHbs+b+fz2mb+mx1d8SZYtn2eg1OijxI95vADX9wAXGOJ2cOO83tCBO207Qp0Hm2p5U52n7Qfg0IQ5nMi3PL/AFRz6+Pd61mWHtPeuD3ndm82Zj9TVg2RgSx5Pdf4/wAy+PEo5r8JY7/aHF4iBblQSAANTJ2BZvsBGtH+A4IuVXMzpbYkuxJ9re1DMJ/8aCVUczmPIEjOF8DeBa1Gf/uOp/LMGzZYbsSIe4NvdUzJW9YArYyqAsAAaD3RyAjkBW7SaYqfUI/CZ9XqAfkX841+GhiI/YNN423GhUgctN29dwP1qbjOIiTqAKH3MYrggnlpJ0nl5dPU10gT2ZzuAaEgX02y76HTqKOLFy3rqGXX1GtAkvgMJGsTBnbaeU0Y4fjVcFQoXLGgj4j1quTnkSxHvMmxPDhau3EMShI15jUCPiPjPKrP2AxoW41rk4zAeK7/ABH0pj+I1g27gurtcXKdBuv00+lVHhXEmtYi3en3WBPls3ymt1erh/nYiT7zaeIYYXbb2zsykfEVgmNUozoTDISreEHUGSAJgj1rfTckAg6HpznpVB7Zdl4c4oBnEybf5VeILhRpJESddvCud/UNhQkC43Giu4BNSg4bB57ZecpU5dT3QpAys25gnSeUfFvB3raaktnIiADppOXTfYa/SifFHCozM265QI3zR3YE5jOvpULh/CMReYKmZSY7o7zbaljML8YqcWVtQlr+c05Ma4WAMfwtq7ede6RbzrmCgkdYd41JAOmgPSj+BwzAF3VUYkgoEI05GSBm8eU7EzAmW+z/ABDh+Fa5bvh0BzPYidCZZlY/m5kAievOoeJ7SYS+kuLtm9GjKcyN0BKkE+o0016mNnU9cfbmXIx13z94D7VlfZKCDn9qxmWjJH8uQdAYN3YbMsyRVYwU+0WN5FS+L8Ra7AOyiBzPiSeZ+m3iYmCWXWdp19KYfqmcm4Y7S4Nbd6RoHAcCNO9Gm8iO98B1qxdgcUVvKswH0PQwCRPrFc/xAw8W8Kf9J2JjTL186Z7LE57fgy/DMI+cmrdZBE/ViM0u+YytzVlP/wAh9iavKGQKomIPcbzHzNXqz7o8hVNV4kaI8GDeNKcugms6x4GfvoSuswddjG/jFX/j924BC7RWc8WuXJM1bTDiaH7i4U1sEG4mbU6BiDtvtEct+VXnDYq0bEhYUfl8azrBXNdRrpBnQdZ01qx4bEHLB0Bgn00/fpTcqXLQ5guH+2BcnIgMdSY5D9aInBWEUTJmdS+vnuB/mnsDbULbtkAjID6kSfnUTGWe6Zg9PA7r6cj4E1hzvk2nZ34lkVb5jK4cjvW9ehnX9+VM3F3LIVP5mAMevQ+NRMRjDbClI16jQiOYon+Je4FcLDRrB36eVeXyaxc6/wDMLI9uD+R6/IidAYTjNoeIzZW5bUG20GCWESW0zaekaUXweJ9oisRuBI8edD8BinuqQQBetytxRyBnIQTuCuvX5ipXDwbVktc0A2HgBAGvjXS04CKAp4qLc333JgtE+HpXOMuLbtsxeNNyRp4+g19KpnaHtYbSNdZiFXZV5kmABG+v3OwqN/1b8QgZXDoRoRtvr6zvz0qTnAFgRq6ck0TKzxO1fxeJY+zYA6IIMBIBAGm8EHLuNiO6Yt3ZfsytqTcIzMIKT+Xch2E93aVXQxBLAxTvC77FnugD2iAaARnXXMh65o0PJoO0gmrXELMK6jMTqp2ENBBk0oDLkdRjHB7P86/OVzNssNHeKhn9kiAA5W20AXQT/pECkuDtqPaPNwzr3oA8QN45a1GxWMdrbuCMxfJI2yryWepM+lCZ/wCa1Z9WyH0x4i8OmVhvk9scrE5LSKvIlQT8xRXhd4RBKBt+6oEAdSBFVs3wNzUfG4Zr9vLZvLN4RlzFSLcj2pHMnKcoI2Lg8qz6fflfuOzKqLG8HiziHuYn8txv5f8A6KSLUeDCbkdbponwoEX4B0ymY25RUQYG9aEG2QAOQ0j02rvhGJAuEkEHKZ+Pyr0RQBNqzjMSbuEe03BhirJtzlYGVPQjr4VkeI4LdW+MO6EOxAA5GTuD08a2q3ilPOvL2EtuyOygshlTzBIj6VXHmbGCIthfU8s2MqqF2AAEdANK7K6a6zy/zTsHlHrURMQCSNyCJAPhI9NtKTIPEqXEexAfEC6Hyo24j3P7B/q8dqs+Gw9jCWzGW2g1Z2IHq7HnU1mkaj05/E1nP8Qez9zOMQWe7ZERZZjlQ8zA2TmY189qBuICL1+kcmRRZb9Zf8BjLWItB0YPbYETrqBKkEHUGQRrWGcd4Q1rE3bKicrGOuUyV08pH+2rl2A7Rn8S2HciLgBSFCqtxFAyBRoqsmUADYoo3Y0//EzCC3ct4ldyCrj+3UN6fY9afiBRypl3p0uZLcqbwC3mv2x1IHx/c1HxZliepn41Z/4bcN9piQ5iEPPnodvLT4igD5otjSGWL+JlqLeHXmM3wgSfpQ/stZ79v+8fv4T8aJ/xDuTct29DudtRLcuh0+dP9kcKS4JOiLoPE6D0EmrHlxFKawmWi4s5V/qdP/tP2q9KNBVO4amfE2xyWWPpoPqfhVypWqPIEvo1pCYP4uBl1rNuObmBWo4u0CNapHaDC7wtRp2ox+QSo2Cg0Yk84EDWOp157Cj+GuBlyqO6oLGW65R3ZOp2mByqvX7TBtF1nSiOBxRPvRqSYURE+m32rawsQEv+Db2iW7g3AhvNdP8AmmMUyqjKC7O+kHYdToNPDxiofBr5CNDQQd/Bh8CJBp+5xIscq21Zo3108TrprXN1GHIyMqGjLK6qwJnj4I5dAHI1AiOXKdfkNq7tXWVdfePIbKKZs5pylyzc9TA9BRfh+HttOskDXfn9fSuTi+EBcgdz8o8C+f18R76ssu0dwFggLBYoABcZTcOsmGksSTMxPxoBh+NPeD+0MFXuCMxIypfu2g0H3ZNt9OUVZeLoqg5jmOsKOQ+31qk8S4S1nE3rhDlbkOAiyWzZGGgGkkOCTA7w1G9a/iKqFBHHtLfD2YkgzjtBxO5h1W/bRLkEqyuJUrdtug2/1MB61XuxntrSkOCFdQQpEQU7uYdZET/aKN2cd7YR7GLXuvmykSwbKuUE6EggzsYEa6ScFgMzhBIEbyTA3gSdNa5pdlT0yJ1xiHqbyYU4B7a5eQWzl11J92BqZ6mBt9KlcO4VYuW7WVGtyoIa07qxkaMQDlbTWGBFTuz/AAlrhDljbtWyVUr79wr3Wg/kXcTuTMRAJsWHt2kULaUKogCOg0AJOtacCsi3dTBqXV3qpUu0HDTh8K0X79xtcgZkUBm/MTZto3TSflJoHa4jcc5Et3M0gRBzTymeZ8as3aPiOIR/Zr3EKyHT3mEgMJ/8cEgEjXvrBWhAxcgWyDbQLlD2WKXV8VZSOe67GTz1pOcozU0bp1dVsC4xfTEWTF8Kmf3bUB71yOSKGA56sTlXdiBUrAcIKIGuQLrBRAMi3bSclsMQCwBLEt+ZmdoAIAHYAXMOzXP5V2d5zK7gHTNduOxLDeGgSfeA1J/D40XCxmGBhlIhlPQruNCPQgiQa6+gTAtnGbnO1j5mIDip6mOv29A5I6HUfOpPD+Me0zZ7KmI1H/NRcZAXmdxpPISfA1W+IcZNu09tdHZtTzAGkajc10Nm7oczGSQIV7QdpcOkpazo/wDUsMAempoj2T7RJiQVmLijUbSP6h4fSsnZSx0p3hPFvw19Lg2DAN4rPe8xz8YFNyYFCfeKJN3Nxe5Iga+X7/cGs/4jxu5ZN19AWkElGdIzdwwp7rgEiTpAk7RV/wALcDqHGoYSPL7frNZL2xY/irlod1VbU85bUROgMH9zWLGoYkGMRqMd4P22v2QvtH9voJDQCunuhp1HiZ66VbeD9t8LfnPNsnSLg7pH9wlfvWbOoAICgESJk79SOdRfbkgHQSBAgnTlz6fs1sOmU9RJPmW3tT2ZOGf8VhTNoglssEoSQQynWFkTI92ByrvHccGNsKHyi9b35BpHvDwPTkfCJq2B4vesz7O4wB3UmVPmp0qO2K7+YKF8BOXXcQeR6foKuMBUfN48yEyEEjxIPEMCVOnu9ek8jWnfw74dksB2glgAukEKGYxtrJJPP7Cl+2DjSQCNZ+Y8f2atGE7QOLWQIokZVAB7pIgGSdfh8KqUANiDlmWhI/F/52LdtCo0HoSNems6VZ+AYbLbLRq2voNv19aD8M4ZGVPV/BRH12q0MDlCL7znKoHpr4ACT6UtB2xlcx6xiGOzFjM1y9yY5V8h5+M1Yqj8PwotW1QbAVIrBkbcxM6ONNqgTxhQPjKaGFk0dpq6mm1QpoyxFzKeK4G5JJEChxxGVl0ACgCBzI3aTPeJ59NKv3HcCTNUjE8PYvlAJJIAHU8q6CZARzKIpLVCfDMU7u2buhpzGI3MzGw1j0NH8CTly2xAnVudUoM6tlgwNCCIgg/sepqy8P46FtlWWWB06QBoDHTb9zUsDXUrkxjfamxCpAQTIUTqfHxO5NNrj2M5JCgQW1kyOXjUXC/z3Lt/2lBmTooIMARHeGnwp+1/OYRC2kkKDsTGpNL/ABiwlcx7DYfuG4/+2d/P9PjURsjELcFwxCqViRJ0BB38I5CiV+5MKdekRrXFkAYi2sbEk89cpP2FIz40yIQ4uMxM+NwVNSvY3s9cDFUViXdcwVWAb2ZzK7lhkEEDUNPLUaEjhez72xLXhnOwtoITxzXAc3/sFWm7ckDxP+KYKgVwjjVeuZ1DndhUk4a2EtJbGyqqjxhQJpnLTbYjKCZ0HL7CNar79pbgch7AtqGC95mZu8G1ZVACgRyZt6azgkE8RSox6Ed7WY3Diy1t7yLeUe0tpm/mHQiMglsrDMsxoddxVRSzeMFQpQwfa5lFvKfzM0wsc9aj9q+M3rjIj27JUswVraMXACliVdzmCnKJGUU52b4eGLXWtjUjKSokkTJE6xBUehq6YF1Dhf3jfUbToT59p7jrTa+zb21se+4SJHSzJOfrOgMAKTOnWB0S03tA1wuqooaT7LKc8gTFuSCDIAYAD3jNt4ThUdir7idPIbVPwPCbLXNNCCCR1I13/wA1vGjxYmDJYr95ibWO6lX5uU7GYsAiYIEEiYkBtV8/L7VVuIX87E7CTAHIEkx86unaTg9r2zLbugNM5SOfTpVK4rgb1onOm/5o03nSNBXXxEHkTG3EZt1AOCEjMRAI9f0HOvRnYhQCSdAANT4ACiF/gd22CbuVGC5spYZ4kSSokjfnTmA6MXRPU0XsZxtbqG0PetxpMiDzkcp9aqv8Q8GRi1I/OqE+atBPwignZvixw+IS5MicreKtv9j6VcP4lg+ztuOTQT5iY+R+JrEcfp5eOjAHyfEoPFH1cjmYHmzQD86hN4bU894HbWN+m2g8+foK9sYYsRO2hjMAxBOw8T5eO1b0ruKaMvbIiREiR4gkifkamcI4c964qqmaCCRMCAdZP5R40c4T2Ne4czSlvcAjvkctPv1nSr7geDizaUIIXUDWSY1+ZJqmTIo4itx/tkHh/Brdm2bYUEN706yOk9B/zuaH4Hg6hjd7xVdFmO8Z3AAA8OtHsPhmvsVUxbB7zD83VV8OtEcThdQFGi7D71iyOL2/rH4UYAuYNwWFyAlveOrfp6Ue4BgNfbMNYhfAT9TXGA4WWILbD51YVUAQKRmy8bRH4MFHe09pUqVZZrirwivaVEJA4hZlTQTCcNVW9p7pGuY7L8dJP6VZbyTQ7EYCd9eg5U1HoVEupJlJ7V4vO3c2AILEatMEyTrGgj160E4TZa6+UaASXc+6o6/Wrhxjg4iW9BVVx94qPZWhC/mPNj+nhW7GwK0JAteDCmKx4YC3bkWVPq5B1YnrqPiKn8HxixkPPbzj7/PTpQ3hfDEOHe47MXVXIURCwpILaTrHhQjDXS1wLMAGS3RRufhRtBsCTuUipe7TZAbnPZJ69fSmbamfaj3lMn0P350HXtFae5DyqjRSAYjxG49J8qsllFKZ1IYEaFSCvoRpSmWhz5lGJB4gP/rwstdt3p/D54tuo1RXVWCOm7qC2UFJIgDLpNTV4vh2XXFW55Anv+RX3viKrF/jbWbtwAhLuZgcyqWC6ZQhI7o592JBBM70rFi9fcnK5n87SCD1AnUxty567HivTPtxr9p10wbU3OwEO/8AVbDSDicvKXt3EAIbQhrqqjGREZhVduOXe6Mt5oZ1EKoBKsRJN5gYkcgT51bcHwEhQAukRU/CdnVUQqhR0AAGpk6DxrUdFhu2MzrqsiWFlN4Rwy4SDcjScqKdi0AsSRvAgaaS2pmjTWI2FWexwdQakpw5ByrSjJjXanUQ5bI25u5T0tuGDARHh4f5o1bzMfaqOk9QfLn5+NG/wadK5bDAarUnNcW2K5nfaTg5a6zIDqZEf6tY9PtUXC4y7aGS+ouJ0O4/WtAv4ImPL7mhnFOCqVJMU5c4Iow9OpBscCtJaN3CqqX7q90tuikfln3SetUnAdn8QuJZLqOWdLqkkEyWttBzbHWKuXGLJmwUmRaAEdRoatHZ6872gbghgY1305npVvVZFJ7uV2gtUyzB9mGwlk4rEoWuR/JsjXvn3S8bwdYFEcJgL2L4eyMCrIynM4yyJGduukt51f8AtLh3a1NtFdlMwRPIyQOZqtdn7zs943CSFstM7DbSOVAyHIu89iKcAMFlUwPYoaF70mSSEB3jSCRryE+Aqx4Lhdi0B7O2AQIDbnfx9a8wmJCAd6DyA97TaANaJWMFduqMilJJlnGoA2ygHWdd+grQz7RyZhpsh4Ea9pBA5sQAOZM6Zec13huFXbxhwbdsEyD7zdZ/pFHOHdn0tnMxLv8A1N9ulGFSKx5NT4WbMWkrl5GwuCVFCgQBTow46U9SrJZm2hPFUCvaVKokxUqVKiEVKlXlEJ4a5IruKRohBuOweehd/gKqsxLGrLFN3SDzFXGQiRtBmf4zgzAmJk9Oh5eVCn4S4kdd/wBK0o4Eanc1HvcPEeNOXUNKnGo6mf4fhGup+VWThOCNs9xiJ300PmDoaJ/gwNqk4awBqal81iAWM4bg6TJEnqRrRaxhFXYCmTdC06MTSGYmWkkLXtRlxM8q9S/Jiq1JkilTJuGlBqIR3NXJauLbiY5106SKDCMXrtCscLh2Ejzojctke9qK9sWxyPoalGMggSsY32/s8qrlA2ho33ExJpxHv2MMtqyF9s0s7HUJm5Dq0RVjca6ikljwkmm+rxRlPTHcqmIONF1HtXDLqsrEjMBDb6ASJ9aL8SRTba0YF26B7QoDGnLwn9aJG02wbKPAAV3YwCjXc9ak5QaMkJUg8E4VbtqIRQesUaC1yiRXdKZixsywAHUVKlSqsmKlSpUQipUqVEIqVKlRCKlSpUQirxhNe0qIRlbAHU+Zpu4hnRR51KpVG0S273jBtE7mm2uqukyalmuBaG8VKgCVMYCZuUUkwmskzUqlUVJBqD8YCD3RJp7D4cx3t6kxXtFcyS3FVOFtAV4bWtOUqmVjOIXSnLY0FdUqIRtresinKVKiEUUzcw4PgaepUQjaJXptiu6VEJ4RSAr2lRCKlSpUQipUqVEIqVKlRCKlSpUQipUqVEJ//9k="],
       "videos": ["https://www.youtube.com/watch?v=LJ6eRZcymmk"] 
    },
}
# ======================
# 유틸
# ======================
def load_pil_from_bytes(b: bytes) -> Image.Image:
    pil = Image.open(BytesIO(b))
    pil = ImageOps.exif_transpose(pil)
    if pil.mode != "RGB": pil = pil.convert("RGB")
    return pil

def yt_id_from_url(url: str) -> str | None:
    if not url: return None
    pats = [r"(?:v=|/)([0-9A-Za-z_-]{11})(?:\?|&|/|$)", r"youtu\.be/([0-9A-Za-z_-]{11})"]
    for p in pats:
        m = re.search(p, url)
        if m: return m.group(1)
    return None

def yt_thumb(url: str) -> str | None:
    vid = yt_id_from_url(url)
    return f"https://img.youtube.com/vi/{vid}/hqdefault.jpg" if vid else None

def pick_top3(lst):
    return [x for x in lst if isinstance(x, str) and x.strip()][:3]

def get_content_for_label(label: str):
    """라벨명으로 콘텐츠 반환 (texts, images, videos). 없으면 빈 리스트."""
    cfg = CONTENT_BY_LABEL.get(label, {})
    return (
        pick_top3(cfg.get("texts", [])),
        pick_top3(cfg.get("images", [])),
        pick_top3(cfg.get("videos", [])),
    )

# ======================
# 입력(카메라/업로드)
# ======================
tab_cam, tab_file = st.tabs(["📷 카메라로 촬영", "📁 파일 업로드"])
new_bytes = None

with tab_cam:
    cam = st.camera_input("카메라 스냅샷", label_visibility="collapsed")
    if cam is not None:
        new_bytes = cam.getvalue()

with tab_file:
    f = st.file_uploader("이미지를 업로드하세요 (jpg, png, jpeg, webp, tiff)",
                         type=["jpg","png","jpeg","webp","tiff"])
    if f is not None:
        new_bytes = f.getvalue()

if new_bytes:
    st.session_state.img_bytes = new_bytes

# ======================
# 예측 & 레이아웃
# ======================
if st.session_state.img_bytes:
    top_l, top_r = st.columns([1, 1], vertical_alignment="center")

    pil_img = load_pil_from_bytes(st.session_state.img_bytes)
    with top_l:
        st.image(pil_img, caption="입력 이미지", use_container_width=True)

    with st.spinner("🧠 분석 중..."):
        pred, pred_idx, probs = learner.predict(PILImage.create(np.array(pil_img)))
        st.session_state.last_prediction = str(pred)

    with top_r:
        st.markdown(
            f"""
            <div class="prediction-box">
                <span style="font-size:1.0rem;color:#555;">예측 결과:</span>
                <h2>{st.session_state.last_prediction}</h2>
                <div class="helper">오른쪽 패널에서 예측 라벨의 콘텐츠가 표시됩니다.</div>
            </div>
            """, unsafe_allow_html=True
        )

    left, right = st.columns([1,1], vertical_alignment="top")

    # 왼쪽: 확률 막대
    with left:
        st.subheader("상세 예측 확률")
        prob_list = sorted(
            [(labels[i], float(probs[i])) for i in range(len(labels))],
            key=lambda x: x[1], reverse=True
        )
        for lbl, p in prob_list:
            pct = p * 100
            hi = "highlight" if lbl == st.session_state.last_prediction else ""
            st.markdown(
                f"""
                <div class="prob-card">
                  <div style="display:flex;justify-content:space-between;margin-bottom:6px;">
                    <strong>{lbl}</strong><span>{pct:.2f}%</span>
                  </div>
                  <div class="prob-bar-bg">
                    <div class="prob-bar-fg {hi}" style="width:{pct:.4f}%;"></div>
                  </div>
                </div>
                """, unsafe_allow_html=True
            )

    # 오른쪽: 정보 패널 (예측 라벨 기본, 다른 라벨로 바꿔보기 가능)
    with right:
        st.subheader("라벨별 고정 콘텐츠")
        default_idx = labels.index(st.session_state.last_prediction) if st.session_state.last_prediction in labels else 0
        info_label = st.selectbox("표시할 라벨 선택", options=labels, index=default_idx)

        texts, images, videos = get_content_for_label(info_label)

        if not any([texts, images, videos]):
            st.info(f"라벨 `{info_label}`에 대한 콘텐츠가 아직 없습니다. 코드의 CONTENT_BY_LABEL에 추가하세요.")
        else:
            # 텍스트
            if texts:
                st.markdown('<div class="info-grid">', unsafe_allow_html=True)
                for t in texts:
                    st.markdown(f"""
                    <div class="card" style="grid-column:span 12;">
                      <h4>텍스트</h4>
                      <div>{t}</div>
                    </div>
                    """, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

            # 이미지(최대 3, 3열)
            if images:
                st.markdown('<div class="info-grid">', unsafe_allow_html=True)
                for url in images[:3]:
                    st.markdown(f"""
                    <div class="card" style="grid-column:span 4;">
                      <h4>이미지</h4>
                      <img src="{url}" class="thumb" />
                    </div>
                    """, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

            # 동영상(유튜브 썸네일)
            if videos:
                st.markdown('<div class="info-grid">', unsafe_allow_html=True)
                for v in videos[:3]:
                    thumb = yt_thumb(v)
                    if thumb:
                        st.markdown(f"""
                        <div class="card" style="grid-column:span 6;">
                          <h4>동영상</h4>
                          <a href="{v}" target="_blank" class="thumb-wrap">
                            <img src="{thumb}" class="thumb"/>
                            <div class="play"></div>
                          </a>
                          <div class="helper">{v}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="card" style="grid-column:span 6;">
                          <h4>동영상</h4>
                          <a href="{v}" target="_blank">{v}</a>
                        </div>
                        """, unsafe_allow_html=True)
else:
    st.info("카메라로 촬영하거나 파일을 업로드하면 분석 결과와 라벨별 콘텐츠가 표시됩니다.")
