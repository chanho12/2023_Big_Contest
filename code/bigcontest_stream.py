import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import streamlit as st
from streamlit_jupyter import StreamlitPatcher, tqdm
from sklearn.metrics.pairwise import cosine_similarity
import streamlit.components.v1 as components
import math
import pyproj
from sklearn.cluster import KMeans

import matplotlib
import seaborn as sns
import folium
from folium.plugins import MarkerCluster

from streamlit_folium import st_folium

#plt.style.use('seaborn-whitegrid')
#matplotlib.rcParams['font.family'] = 'MalgunGothic'
matplotlib.rcParams['font.size'] = 10
pd.options.display.float_format = '{:.10f}'.format


def Make_Cosine_sim(df_risky_0, df_risky_1):
    for number in df_risky_1.index:
        Cosine = []
        Number = []
        data = df_risky_1.loc[number][4:22]
        for j in df_risky_0.index:
            if (df_risky_1.loc[number][1] == df_risky_0.loc[j][1]) and (df_risky_1.loc[number][0] != df_risky_0.loc[j][0] ) :
                Cos = cosine_similarity(data.values.reshape(1, -1), df_risky_0.loc[j][4: -1].values.reshape(1, -1))[0][0]
                Cosine.append(Cos)
                Number.append(j)
        diction = {'number' : Number, 'cosine' : Cosine}
        dataframe = pd.DataFrame(diction, columns = ['number','cosine']).sort_values(by = 'cosine', ascending = False)
        
        
        if len(dataframe) > 0:
            best_number = dataframe.iloc[0][0]
            best_cos = dataframe.iloc[0][1]
        else:
            best_number = None
            best_cos = float('nan')
        
        
        df_risky_1.loc[number, 'cosine'] = best_cos
        if best_number is not None:
            df_risky_1.loc[number, 'similarity'] = df_risky_0.loc[best_number][0] + df_risky_0.loc[best_number][1]
        else:
            df_risky_1.loc[number, 'similarity'] = float('nan')
            
def Recsys_Where_to_Cluster(persona): ## 페르소나 데이터가 어느 클러스터에 가장 적합한지.
    Cluster_list = [Recsys0, Recsys1, Recsys2, Recsys3, Recsys4]
    sim_list = []
    for cluster in Cluster_list:
        cosine = cosine_similarity(persona.drop('risky_pct')[2:].values.reshape(1, -1), cluster.iloc[:, 4: -1].mean().values.reshape(1, -1))[0][0]
        sim_list.append(cosine)
    number = np.argmax(sim_list)
    return number

def Recsys_Who_to_Sosang(persona, number): ## 페르소나가 그 클러스터에서 어느 데이터와 가장 비슷한지.
    Cluster_list = [Recsys0, Recsys1, Recsys2, Recsys3, Recsys4]
    Recsys_cluster = Cluster_list[number]
    
    Recsys_cluster = Recsys_cluster[Recsys_cluster['risky_pct'] == 0]
    
    Cosine = []
    Number = []
    for idx in Recsys_cluster.index:
        if (Recsys_cluster.loc[idx]['class_2'] == persona['class_2']) and (Recsys_cluster.loc[idx]['trdar_nm'] != persona['trdar_nm']):
            cosine = cosine_similarity(Recsys_cluster.loc[idx][4:-1].values.reshape(1, -1), persona.drop('risky_pct')[2:].values.reshape(1, -1))
            Number.append(idx)
            Cosine.append(cosine)
    diction = {'number' : Number, 'cosine' : Cosine}
    dataframe = pd.DataFrame(diction, columns = ['number','cosine']).sort_values(by = 'cosine', ascending = False)

    if len(dataframe) > 0:
        best_number = dataframe.iloc[0][0]
        best_cos = dataframe.iloc[0][1]
    else:
        best_number = None
        best_cos = float('nan')
    
    return Recsys_cluster.loc[best_number]


cl1, lc2, lc3  = st.tabs(['추천시스템','유통입지','기타 시각화'])

with cl1 :
    cl1.title('상권 유사도 추천') #제목 수정하고 싶으면 여기




    Recsys_data = pd.read_csv("./recommender_system(without_logistics).csv", index_col = 0)
    Recsys = Recsys_data.copy()
    Recsys = Recsys.fillna(0)

    ### 클러스터 0
    Recsys0 = Recsys[Recsys['Cluster_Labels'] == 0.0]
    Recsys0_risky_1 = Recsys0[Recsys0['risky_pct'] == 1]
    Recsys0_risky_0 = Recsys0[Recsys0['risky_pct'] == 0]
    Make_Cosine_sim(Recsys0_risky_0, Recsys0_risky_1)


    ### 클러스터 1
    Recsys1 = Recsys[Recsys['Cluster_Labels'] == 1.0]
    Recsys1_risky_1 = Recsys1[Recsys1['risky_pct'] == 1]
    Recsys1_risky_0 = Recsys1[Recsys1['risky_pct'] == 0]
    Make_Cosine_sim(Recsys1_risky_0, Recsys1_risky_1)


    ### 클러스터 2
    Recsys2 = Recsys[Recsys['Cluster_Labels'] == 2.0]
    Recsys2_risky_1 = Recsys2[Recsys2['risky_pct'] == 1]
    Recsys2_risky_0 = Recsys2[Recsys2['risky_pct'] == 0]
    Make_Cosine_sim(Recsys2_risky_0, Recsys2_risky_1)


    ### 클러스터 3
    Recsys3 = Recsys[Recsys['Cluster_Labels'] == 3.0]
    Recsys3_risky_1 = Recsys3[Recsys3['risky_pct'] == 1]
    Recsys3_risky_0 = Recsys3[Recsys3['risky_pct'] == 0]
    Make_Cosine_sim(Recsys3_risky_0, Recsys3_risky_1)


    ### 클러스터 4
    Recsys4 = Recsys[Recsys['Cluster_Labels'] == 4.0]
    Recsys4_risky_1 = Recsys4[Recsys4['risky_pct'] == 1]
    Recsys4_risky_0 = Recsys4[Recsys4['risky_pct'] == 0]
    Make_Cosine_sim(Recsys4_risky_0, Recsys4_risky_1)    



    persona = {
        'trdar_nm' : '영등포구',
        'class_2' : '술집',
        'risky_pct' : 0,
        'age_avg': 46,
        'age_std': 8.5938292,
        'duration_avg': 5.392838192,
        'duration_std': 3.30942890,
        'franchise_pct': 0.22112122,
        'franchise_std': 0.443432,
        'business_square_size_avg': 76.43423,
        'business_square_size_std': 48.12434322,
        'rental_fee_avg': 1.25242323,
        'rental_fee_std': 0.232121212,
        'employee_cnt_avg': 1.4834832,
        'employee_cnt_std': 3.232121212,
        'rental_deposit_avg': 9.232121212,
        'rental_deposit_std': 6.123123,
        'customer_cnt_avg': 22.23234234234,
        'customer_cnt_std': 32.43434323,
        'new_customer_cnt_avg': 3.2323212,
        'new_customer_cnt_std': 3.34123123
    } #default

    persona = pd.Series(persona)



    Rec_number = Recsys_Where_to_Cluster(persona)

    c1, c2 = cl1.columns(2)
    with c1 :
        op_trdar_nm = cl1.selectbox(
                '상권 구분',
                Recsys.trdar_nm.unique())
        op_class2 = cl1.selectbox(
                '업종 구분',
                Recsys.class_2.unique())
        franchise = cl1.radio('프랜차이즈 여부', ('예','아니오')) #
        op_fr = 1 if franchise == '예' else 0
        in_age = cl1.number_input('업주 나이', min_value=0,max_value=100, step=1) #
        in_dur = cl1.number_input('업력', min_value=0,max_value=100, step=1) #
        in_size = cl1.number_input('평수', min_value=0,max_value=100, step=1)
    with c2 :
        in_rent = cl1.number_input('월세', min_value=0,max_value=100, step=1) 
        in_dep = cl1.number_input('보증금', min_value=0,max_value=100, step=1)
        in_emp = cl1.number_input('종업원 수', min_value=0,max_value=100, step=1)
        in_cust = cl1.number_input('손님 수', min_value=0,max_value=100, step=1)
        in_ncust = cl1.number_input('새로운 손님 수', min_value=0,max_value=100, step=1)

    if Recsys.loc[(Recsys.trdar_nm==op_trdar_nm)&(Recsys.class_2==op_class2)&(Recsys.risky_pct==0)].empty == False:
        x= Recsys.loc[(Recsys.trdar_nm==op_trdar_nm)&(Recsys.class_2==op_class2)&(Recsys.risky_pct==0)]
        cl1.write(op_trdar_nm+ '상권에서 ' +op_class2+ '업종을 운영하고 있습니다.')
        new_cnt=x['store_cnt'].values[0] + 1
        #나이
        new_age_avg=(( x['age_avg'].values[0] * x['store_cnt'].values[0]) + in_age)/new_cnt
        sum_of_age_std = (x['age_std'].values[0]**2 * new_cnt) + ((in_age - new_age_avg) ** 2)
        new_age_dev = math.sqrt(sum_of_age_std / (new_cnt - 1))
        #업력
        new_duration_avg=(( x['duration_avg'].values[0] * x['store_cnt'].values[0]) + in_dur)/new_cnt
        sum_of_duration_std = (x['duration_std'].values[0]**2 * new_cnt) + ((in_dur - new_duration_avg) ** 2)
        new_duration_dev = math.sqrt(sum_of_duration_std / (new_cnt - 1))
        #평수
        new_business_square_size_avg=(( x['business_square_size_avg'].values[0] * x['store_cnt'].values[0]) + in_size)/new_cnt
        sum_of_business_square_size_std = (x['business_square_size_std'].values[0]**2 * new_cnt) + ((in_size - new_business_square_size_avg) ** 2)
        new_business_square_size_dev = math.sqrt(sum_of_business_square_size_std / (new_cnt - 1))
        #프랜차이즈
        new_franchise_pct=(( x['franchise_pct'].values[0] * x['store_cnt'].values[0]) + op_fr)/new_cnt
        sum_of_franchise_pct_std = (x['franchise_std'].values[0]**2 * new_cnt) + ((op_fr - new_franchise_pct) ** 2)
        new_franchise_pct_dev = math.sqrt(sum_of_franchise_pct_std / (new_cnt - 1))
        #월세
        new_rental_fee_avg=(( x['rental_fee_avg'].values[0] * x['store_cnt'].values[0]) + in_rent)/new_cnt
        sum_of_rental_fee_std = (x['rental_fee_std'].values[0]**2 * new_cnt) + ((in_rent - new_rental_fee_avg) ** 2)
        new_rental_fee_dev = math.sqrt(sum_of_rental_fee_std / (new_cnt - 1))
        #보증금
        new_rental_deposit_avg=(( x['rental_deposit_avg'].values[0] * x['store_cnt'].values[0]) + in_dep)/new_cnt
        sum_of_rental_deposit_std = (x['rental_deposit_std'].values[0]**2 * new_cnt) + ((in_dep - new_rental_deposit_avg) ** 2)
        new_rental_deposit_dev = math.sqrt(sum_of_rental_deposit_std / (new_cnt - 1))
        #종업원
        new_employee_cnt_avg=(( x['employee_cnt_avg'].values[0] * x['store_cnt'].values[0]) + in_emp)/new_cnt
        sum_of_employee_cnt_std = (x['employee_cnt_std'].values[0]**2 * new_cnt) + ((in_emp - new_employee_cnt_avg) ** 2)
        new_employee_cnt_dev = math.sqrt(sum_of_employee_cnt_std / (new_cnt - 1))
        #손님 수
        new_customer_cnt_avg=(( x['customer_cnt_avg'].values[0] * x['store_cnt'].values[0]) + in_cust)/new_cnt
        sum_of_customer_cnt_std = (x['customer_cnt_std'].values[0]**2 * new_cnt) + ((in_cust - new_customer_cnt_avg) ** 2)
        new_customer_cnt_dev = math.sqrt(sum_of_customer_cnt_std / (new_cnt - 1))
        #새로운 손님수
        new_new_customer_cnt_avg=(( x['new_customer_cnt_avg'].values[0] * x['store_cnt'].values[0]) + in_ncust)/new_cnt
        sum_of_new_customer_cnt_std = (x['new_customer_cnt_std'].values[0]**2 * new_cnt) + ((in_ncust - new_new_customer_cnt_avg) ** 2)
        new_new_customer_cnt_dev = math.sqrt(sum_of_new_customer_cnt_std / (new_cnt - 1))

        new_persona = {
        'trdar_nm' : op_trdar_nm,
        'class_2' : op_class2,
        'risky_pct' : 0,
        'age_avg': new_age_avg,
        'age_std': new_age_dev,
        'duration_avg': new_duration_avg,
        'duration_std': new_duration_dev,
        'franchise_pct': new_franchise_pct,
        'franchise_std': new_franchise_pct_dev,
        'business_square_size_avg':new_business_square_size_avg,
        'business_square_size_std': new_business_square_size_dev,
        'rental_fee_avg': new_rental_fee_avg,
        'rental_fee_std': new_rental_fee_dev,
        'employee_cnt_avg': new_employee_cnt_avg,
        'employee_cnt_std': new_employee_cnt_dev,
        'rental_deposit_avg': new_rental_deposit_avg,
        'rental_deposit_std': new_rental_deposit_dev,
        'customer_cnt_avg': new_customer_cnt_avg,
        'customer_cnt_std': new_customer_cnt_dev,
        'new_customer_cnt_avg': new_new_customer_cnt_avg,
        'new_customer_cnt_std':  new_new_customer_cnt_dev}
        new_persona = pd.Series(new_persona)
        new_Rec_number = Recsys_Where_to_Cluster(new_persona)
        new_trdar = Recsys_Who_to_Sosang(new_persona, new_Rec_number)


        clust_info = {
        0 : {'name' : '나들이 & 데이트 코스',
            '지원제도':'다시서기 4.0 정책 미선정',
            '기타' : '기초교육 제공, 폐업지원 / 재취업 / 창업사관학교 등을 통한 재창업을 지원합니다.'},
        1 : {'name' : '생활 편의 시설',
            '지원제도':'다시서기 4.0 정책 선정\n\n현재 사업장을 운영중이며 채무/신용 불량 등 재무적으로 결격사유가 없고, 사업을 지속하고자 하는 적극적인 의지가 있는 사업자들을 위한 교육/컨설팅/자금/대출 지원 정책',
            '기타' : '안정적인 컨설팅/관리 교육 등의 정책을 추가로 제공합니다.'},
        2 : {'name' : '오피스 상권',
            '지원제도':'다시서기 4.0 정책 선정\n\n현재 사업장을 운영중이며 채무/신용 불량 등 재무적으로 결격사유가 없고, 사업을 지속하고자 하는 적극적인 의지가 있는 사업자들을 위한 교육/컨설팅/자금/대출 지원 정책',
            '기타' : ', 1인 사업장 : 1인사업지원정책\n강한소상공인정책을 맞춤으로 제공하여 지원합니다'},
        3 : {'name' : '주택가 상권',
            '지원제도':'다시서기 4.0 정책 미선정',
            '기타' : '프랜차이즈 업종 지원 / 사회환원 정책을 통해 사회기금을 지원합니다.'},
        4 : {'name' : '대학가 상권',
            '지원제도':'다시서기 4.0 정책 선정\n\n현재 사업장을 운영중이며 채무/신용 불량 등 재무적으로 결격사유가 없고, 사업을 지속하고자 하는 적극적인 의지가 있는 사업자들을 위한 교육/컨설팅/자금/대출 지원 정책',
            '기타' : '상권 협업 정책을 제안합니다.'}
    }

        cl1.subheader(clust_info[new_Rec_number]['name'])
        cl1.info(clust_info[new_Rec_number]['지원제도'])
        cl1.write(clust_info[new_Rec_number]['기타'])

        cl1.write(new_trdar['trdar_nm']+'과 유사합니다.')

    else :
        cl1.write('해당 상권 내 '+ op_trdar_nm+ '업종이 없습니다. 표본을 추가해야 합니다.')
    
with lc3 :
    anova_data1 = pd.read_csv("./data/Anova_class_1_name.csv", index_col = 0)
    anova_data2 = pd.read_csv("./data/Anova_class_2_name.csv", index_col = 0)
    anova_data3 = pd.read_csv("./data/Anova_is_franchise.csv", index_col = 0)
    anova_data4 = pd.read_csv("./data/Anova_is_risky.csv", index_col = 0)
    anova_data5 = pd.read_csv("./data/Anova_quarter.csv", index_col = 0)
    anova_data6 = pd.read_csv("./data/Anova_rental_type.csv", index_col = 0)
    anova_data7 = pd.read_csv("./data/Anova_rental_type_자가_타가(개인).csv", index_col = 0)
    anova_data8 = pd.read_csv("./data/Anova_rental_type_자가_타가(법인).csv", index_col = 0)
    anova_data9 = pd.read_csv("./data/Anova_rental_type_타가(개인)_타가(법인).csv", index_col = 0)
    df_asc_cluster = pd.read_csv('./data/clustered_df.csv', index_col=0)
    
    anova_data = [anova_data1, anova_data2, anova_data3, anova_data4, anova_data5, anova_data6, anova_data7, anova_data8, anova_data9]
    
    sns.set_style("whitegrid")
    lc3.subheader('각 변수별 분산분석 결과')
    for i in anova_data:
    # 그래프 크기 조정
        fig=plt.figure(figsize=(10,6))

        # 데이터 정렬 및 바 차트 그리기
        ax = i.T.sort_index().plot(kind='bar', color=sns.color_palette("pastel"))

        # 레드라인 추가
        ax.axhline(y=0.01, color='red', linestyle='--', label='Threshold = 0.01')

        # 제목 및 라벨 추가
        ax.set_title("ANOVA Data Visualization", fontsize=18)
        ax.set_xlabel("X-axis Label", fontsize=15)
        ax.set_ylabel("Y-axis Label", fontsize=15)

        # 범례 추가
        ax.legend()

        plt.tight_layout()
        st.pyplot(plt.gcf())
    lc3.subheader('상권/업종별 군집화 결과')
    fig=plt.figure(figsize=(40, 8))   
    cluster_label_mapping = {label: i for i, label in enumerate(df_asc_cluster['Cluster_Labels'].unique())}

    # Map the Cluster_Labels column to numeric values
    df_asc_cluster['Cluster_Labels_Num'] = df_asc_cluster['Cluster_Labels'].map(cluster_label_mapping)

    # Pivot the DataFrame for visualization
    pivot_df = df_asc_cluster.pivot(index='class_2', columns='trdar_nm', values='Cluster_Labels_Num')

    # 2개로 분리 
    pivot_df1 = pivot_df.iloc[:, :53]
    pivot_df2 = pivot_df.iloc[:, 53:]

    # Define custom pastel colors for each cluster label
    cluster_colors = {
        0: '#FFB6C1',  # Light Pink
        1: '#FFD700',  # Gold
        2: '#90EE90',  # Light Green
        3: '#87CEEB',  # Sky Blue
        4: '#FFA07A'   # Light Salmon
    }

    # Create a custom colormap with distinct pastel colors for each cluster label
    num_clusters = len(df_asc_cluster['Cluster_Labels_Num'].unique())
    cmap = sns.color_palette([cluster_colors[i] for i in range(num_clusters)])

    # Create a heatmap with custom colormap
    plt.figure(figsize=(40, 8))
    sns.heatmap(pivot_df1, cmap=cmap, annot=True, fmt=".0f", linewidths=.5, cbar_kws={'label': 'Cluster Labels'})
    plt.xlabel('trdar_nm')
    plt.ylabel('class_2')
    
    plt.title('Cluster Label Heatmap')
    st.pyplot(plt.gcf())
    
    fig=plt.figure(figsize=(40, 8))
    
    sns.heatmap(pivot_df2, cmap=cmap, annot=True, fmt=".0f", linewidths=.5, cbar_kws={'label': 'Cluster Labels'})
    plt.xlabel('trdar_nm')
    plt.ylabel('class_2')
    plt.title('Cluster Label Heatmap')
    plt.show()
    st.pyplot(plt.gcf())

        
with lc2 :
    data = pd.read_csv("./data/서울시 유통전문판매업 인허가 정보.csv", index_col = 0 ,encoding = 'cp949')
    df = data.copy()
    df = df[df['상세영업상태명'] == '영업']
    df = df.dropna(subset=['좌표정보(X)']).dropna(subset = ['좌표정보(Y)'])
    
    df['소재지면적'] = pd.to_numeric(df['소재지면적'], errors='coerce')

    kmeans = KMeans(n_clusters=8).fit(df[['좌표정보(X)', '좌표정보(Y)']])
    df['Cluster'] = kmeans.labels_

    def weighted_center(cluster_df):
        x_center = np.sum(cluster_df['좌표정보(X)'] * cluster_df['소재지면적']) / np.sum(cluster_df['소재지면적'])
        y_center = np.sum(cluster_df['좌표정보(Y)'] * cluster_df['소재지면적']) / np.sum(cluster_df['소재지면적'])
        return x_center, y_center

    centers = df.groupby('Cluster').apply(weighted_center)
    fig1=plt.figure(figsize=(10,6))
    plt.scatter(df['좌표정보(X)'], df['좌표정보(Y)'], c=df['Cluster'], cmap='rainbow', s=df['소재지면적'], alpha=0.6)  # 상점 위치
    plt.scatter(centers.apply(lambda x: x[0]), centers.apply(lambda x: x[1]), s=200, c='black', marker='X')  # 중심점
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Shops Clustering with Weighted Centers')
    st.pyplot(plt.gcf())

    folium_data = pd.read_csv("./data/유통전문판매업.csv", index_col = 0 )
    def tm_to_latlon(x, y):
        transformer = pyproj.Transformer.from_crs("EPSG:2097", "EPSG:4326", always_xy=True)
        lon, lat = transformer.transform(x, y)
        return lat, lon
    
    cluster_centers = pd.DataFrame([[i[0], i[1]] for i in centers], columns = ['latitude','longitude'])
    cluster_centers['latitude'], cluster_centers['longitude'] = zip(*cluster_centers.apply(lambda row: tm_to_latlon(row['latitude'], row['longitude']), axis=1))

    colors = ["red", "blue", "green", "purple", "orange", "pink", "black", "gray"]


    # Folium 지도 생성
    port_m = folium.Map(location=[folium_data['latitude'].median(), folium_data['longitude'].median()] , zoom_start=11)


    unique_clusters = folium_data['Cluster'].unique()
    colors_map = plt.cm.rainbow(np.linspace(0, 1, len(unique_clusters)))
    colors = {cluster: matplotlib.colors.to_hex(colors_map[i]) for i, cluster in enumerate(unique_clusters)}

    ### 일반 그래프 그리기

    # 각 점을 지도에 표시
    for idx, row in folium_data.iterrows():
        cluster_color = colors[row['Cluster']]
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            color=cluster_color,
            fill=True,
            fill_color=cluster_color,
            fill_opacity=0.6,
            radius=5  # 원하는 반지름 값으로 조정 가능
        ).add_to(port_m)



    ### 중심 센터들 색칠하기
    for idx, row in cluster_centers.iterrows():
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=10,
            popup=f"Centroid {idx}",
            fill=True,
            color="black",  # 검은색으로 지정
            fill_color="black"  # 검은색으로 지정
        ).add_to(port_m)
        
    st_data = st_folium(port_m, width=725)
