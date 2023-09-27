import pandas as pd
import numpy as np
import random

# définir le nombre d'échantillons
num_samples = 50000

# générer les données psychologiques (utiliser une échelle de 1 à 5)
pss_scores = np.random.randint(low=1, high=5, size=num_samples) # échelle de stress perçu (PSS)
jcq_scores = np.random.randint(low=1, high=5, size=num_samples) # questionnaire Job Content Questionnaire (JCQ)

# générer les données physiologiques
heart_rates = np.random.randint(low=60, high=100, size=num_samples) # fréquence cardiaque
blood_oxygen_levels = np.random.uniform(low=95, high=100, size=num_samples) # taux d'oxygène dans le sang
skin_temperatures = np.random.uniform(low=30, high=35, size=num_samples) # température de la peau

# générer les données de charges de travail
email_volume = np.random.randint(low=20, high=200, size=num_samples) # volume de mails
task_numbers = np.random.randint(low=1, high=10, size=num_samples) # nombre de tâches
workload_perception = np.random.randint(low=1, high=5, size=num_samples) # perception de la charge de travail

# Créer un DataFrame
df = pd.DataFrame({
    'PSS_Score': pss_scores,
    'JCQ_Score': jcq_scores,
    'Heart_Rate': heart_rates,
    'Blood_Oxygen_Level': blood_oxygen_levels,
    'Skin_Temperature': skin_temperatures,
    'Email_Volume': email_volume,
    'Number_of_Tasks': task_numbers,
    'Workload_Perception': workload_perception
})

# Calculer un score de stress en utilisant une combinaison de ces facteurs
df['Stress_Score'] = df['PSS_Score'] + df['JCQ_Score'] + df['Heart_Rate']/100 + (100 - df['Blood_Oxygen_Level'])/10 
+ (37 - df['Skin_Temperature']) + df['Number_of_Tasks']/10 + df['Workload_Perception'] + df['Email_Volume']/100

# Ajouter un peu de bruit aléatoire
df['Stress_Score'] += np.random.normal(scale=0.1, size=num_samples)

# Attribuer les états de stress en fonction du score de stress
df['Stress_State'] = np.where(df['Stress_Score'] > df['Stress_Score'].median(), 'stress', 'non stress')

# Afficher le DataFrame
print(df.head())
# Exporter le DataFrame en .xlsx
df.to_excel('stress_data.xlsx', index=False)
