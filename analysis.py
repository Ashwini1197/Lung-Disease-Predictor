import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('mdata.csv', engine='python', encoding='utf_8_sig')


def analysis():
    
        age1=df[(df.Age>=0) & (df.Age<31)]
        age2=df[(df.Age>=31) & (df.Age<61)]
        age3=df[(df.Age>=61) & (df.Age<91)]




        labels =["0-30","31-60","61-90"]
        df['bins']= pd.cut(df['Age'],bins = [0,30,60,90],labels = labels)
        ef= df.groupby('bins').size()

        subgroup_names=['Asthma','COPD','Restrictive','Normal','Asthma','COPD','Restrictive','Normal','Asthma','COPD','Restrictive','Normal']
        s1name = ['Asthma','COPD','Restrictive','Normal']
        age1_size=age1.groupby(age1['Disease']).size().reset_index(name='counts')
        age2_size=age2.groupby(age2['Disease']).size().reset_index(name='counts')
        age3_size=age3.groupby(age3['Disease']).size().reset_index(name='counts')
        



        a1=age1_size.counts[0]
        a2=age1_size.counts[1]
        a3=age1_size.counts[2]
        a4=age1_size.counts[3]
        b1=age2_size.counts[0]
        b2=age2_size.counts[1]
        b3=age2_size.counts[2]
        b4=age2_size.counts[3]
        c1=age3_size.counts[0]
        c2=age3_size.counts[1]
        c3=age3_size.counts[2]
        c4=age3_size.counts[3]


        #subgroup_size=[37,44,72,64,84,48,28,9,13]
        subgroup_size=[a1,a2,a3,a4,b1,b2,b3,b4,c1,c2,c3,c4]
        #a, b, c=[plt.cm.Blues, plt.cm.Reds, plt.cm.Greens]

        colors = ['#ff6666', '#99ff99', '#66b3ff']
        colors_disease = ['#c2c2f0','#ffb3e6', '#ffcc99','#3c9713','#c2c2f0','#ffb3e6', '#ffcc99','#3c9713', '#c2c2f0','#ffb3e6', '#ffcc99','#3c9713']

        explode = (0.2,0.2,0.2) 
        explode_gender = (0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1)                  
        # First Ring (outside)

        plt.pie(ef, labels=labels, colors=colors,startangle=90,frame=True,explode=explode,radius=4) 


        # Second Ring (Inside)

        plt.pie(subgroup_size,labels=subgroup_size,colors=colors_disease,startangle=90,explode=explode_gender,radius=2)
        centre_circle = plt.Circle((0,0),0.5,color='black', fc='white',linewidth=0)
        patches, texts = plt.pie(subgroup_size, colors=colors_disease, startangle=90)
        plt.legend(patches, s1name, loc="best")

        fig = plt.gcf()
        fig.gca().add_artist(centre_circle)


 
        plt.axis('equal')
        plt.tight_layout()
        plt.title("Distribution of Lung diseases based on Age");
        plt.savefig("C:/Users/ashwini/project/sem1/static/Pie_1.png")
        plt.legend()
        plt.show()
        
        
        
        smoke_yes=df[df.Smoker=='Y']
        smoke_no=df[df.Smoker=='N']
        labels1=['Smoker=yes','Smoker=no']
        sy_size=smoke_yes.groupby(smoke_yes['Disease']).size().reset_index(name='counts')
        ny_size=smoke_no.groupby(smoke_no['Disease']).size().reset_index(name='counts')
        d1=sy_size.counts[0]
        d2=sy_size.counts[1]
        d3=sy_size.counts[2]
        d4=sy_size.counts[3]
        e1=ny_size.counts[0]
        e2=ny_size.counts[1]
        e3=ny_size.counts[2]
        e4=ny_size.counts[3]
        sf=[d1+d2+d3+d4,e1+e2+e3+e4]
        sg_size=[d1,d2,d3,d4,e1,e2,e3,e4]
        colors1 = ['#ed3711', '#3c9713']
        colors_smoker = ['#0ff428','#f4ed16', '#f4509b','#e4c542','#0ff428','#f4ed16', '#f4509b','#e4c542']           
        explode1 = (0.2,0.2) 
        explode_smoker = (0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1)            

        plt.pie(sf, labels=labels1, colors=colors1,startangle=90,frame=True,explode=explode1,radius=3)


        plt.pie(sg_size,labels=sg_size,colors=colors_smoker,startangle=90,explode=explode_smoker,radius=2)
        centre_circle = plt.Circle((0,0),0.5,color='black', fc='white',linewidth=0)
        patches, texts = plt.pie(sg_size, colors=colors_smoker, startangle=90)
        plt.legend(patches, s1name, loc="best")

        fig = plt.gcf()
        fig.gca().add_artist(centre_circle)



        plt.axis('equal')
        plt.tight_layout()
        plt.title("Distribution of Lung diseases based on Smoker or Non-Smoker");
        plt.savefig("C:/Users/ashwini/project/sem1/static/Pie_2.png")
        plt.legend()
        plt.show()
        
        #plt.pie(figsize = (8,8),textprops= {'fontsize' :14})



        female=df[df.Gender=='F']
        male=df[df.Gender=='M']
        labels2=['Female','Male']
        female_size=female.groupby(female['Disease']).size().reset_index(name='counts')
        male_size=male.groupby(male['Disease']).size().reset_index(name='counts')
        f1=female_size.counts[0]
        f2=female_size.counts[1]
        f3=female_size.counts[2]
        f4=female_size.counts[3]
        g1=male_size.counts[0]
        g2=male_size.counts[1]
        g3=male_size.counts[2]
        g4=male_size.counts[3]
        gf=[f1+f2+f3+f4,g1+g2+g3+g4]
        subg_size=[f1,f2,f3,f4,g1,g2,g3,g4]
        colors2 = ['#dc1dc1','#27a4c1']
        colors_gender = ['#42e4c5','#b2e442', '#e4c542','#3c9713','#42e4c5','#b2e442', '#e4c542','#3c9713']           
        explode2 = (0.2,0.2) 
        explode_gender = (0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1)            

        plt.pie(sf, labels=labels2, colors=colors2,startangle=90,frame=True,explode=explode2,radius=3)


        plt.pie(sg_size,labels=subg_size,colors=colors_gender,startangle=90,explode=explode_gender,radius=2)
        centre_circle = plt.Circle((0,0),0.5,color='black', fc='white',linewidth=0)
        patches, texts = plt.pie(subg_size, colors=colors_gender, startangle=90)
        plt.legend(patches, s1name, loc="best")

        fig = plt.gcf()
        fig.gca().add_artist(centre_circle)



        plt.axis('equal')
        plt.tight_layout()
        plt.title("Distribution of Lung diseases based on Gender");
        plt.savefig("C:/Users/ashwini/project/sem1/static/Pie_3.png")
        plt.legend()
        plt.show()
