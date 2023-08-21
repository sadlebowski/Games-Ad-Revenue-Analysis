#!/usr/bin/env python
# coding: utf-8

# # Игры — Анализ рекламных источников

# ## Материалы:
# * [Презентация] (https://disk.yandex.ru/i/YIsHYxdTFQs-Og)
# * [Дашборд] (https://public.tableau.com/app/profile/sadlebowski/viz/_16798385156570/sheet2?publish=yes)

# **Цель исследования:**
# 
# Проанализировать поведения игроков в зависимости от источника перехода.
# - Провести исследовательский анализ данных;
# - Проанализировать влияние источника перехода в игру на поведение пользователя;
# - Проверить статистические гипотезы
#     1. *Проверить гипотезу: время завершения уровня различается в зависимости способа прохождения:*
#         - *через реализацию проекта,*
#         - *через победу над первым игроком.*
#     2. *Сформулировать собственную статистическую гипотезу. Дополнить её нулевой и альтернативной гипотезами. Проверить гипотезу с помощью статистического теста.*
# 
# Обозначить приоритезацию каналов, в какой источник стоит вкладывать больше денег

# **Описание данных:**
# 
# В датасете содержатся данные первых пользователей приложения — когорты пользователей, которые начали пользоваться приложением в период с 4 по 10 мая включительно.
# 
# Датасет *game_actions.csv*:
# 
# - `event_datetime` — время события;
# - `event` — одно из трёх событий:
#     1. `building` — объект построен,
#     2. `finished_stage_1` — первый уровень завершён,
#     3. `project` — проект завершён;
# - `building_type` — один из трёх типов здания:
#     1. `assembly_shop` — сборочный цех,
#     2. `spaceport` — космопорт,
#     3. `research_center` — исследовательский центр;
# - `user_id` — идентификатор пользователя;
# - `project_type` — тип реализованного проекта;
# 
# Датасет *ad_costs.csv*:
# 
# - `day` - день, в который был совершен клик по объявлению
# - `source` - источник трафика
# - `cost` - стоимость кликов
# 
# Датасет user_source.csv содержит колонки:
# 
# - `user_id` - идентификатор пользователя
# - `source` - источников, с которого пришёл пользователь, установивший приложение

# **Ход исследования:**
# 
# **1. Импортировать библиотеки, изучить общую информацию и сделать предобработку данных:**
#     
#    1.1 Проверить таблицы на пропущенные значения, если нужно избавиться от них
#    
#    1.2 Преобразовать данные в подходящий формат
# 
# **2. Исследовательский анализ данных:**
# 
#    2.1 Проанализировать распределение количества пользователей из каждого источника
#    
#    2.2 Проанализировать подходы к игровым стратегиям пользователей:
#    * Распределить пользователей на группы с разными стратегиями игры и посчитать их количество
#    * Посмотреть на количество игроков завершивших уровень
#    * Проанализировать кол-во построек, которое построили игроки в каждой из групп
#    2.3 Проанализировать метрики
# 
# **3. Проверка статистических гипотез:**
#     
#    3.1 Проверить гипотезу: время завершения уровня различается в зависимости способа прохождения:
#     нулевая - через реализацию проекта,
#     альтернативная - через победу над первым игроком.
#     
#    3.2 Проверить гипотезу: влияет ли источник на количество совершённых событий пользователем:
#     нулевая - количестов действий не зависит от источника
#     альтернативная - количество действий зависит от источника
# 
# **4. Выводы и рекомендации:**

# # 1. Импортировать библиотеки, изучить общую информацию и сделать предобработку данных

# In[1]:


#импортируем библиотеки
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from scipy import stats as st
import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('chained_assignment', None)


# In[2]:


#загружаем данные
try:
    game_actions = pd.read_csv('https://code.s3.yandex.net/datasets/game_actions.csv')
    user_source = pd.read_csv('https://code.s3.yandex.net/datasets/user_source.csv')
    ad_costs = pd.read_csv('https://code.s3.yandex.net/datasets/ad_costs.csv')
except:
    game_actions = pd.read_csv('game_actions.csv')
    user_source = pd.read_csv('user_source.csv')
    ad_costs = pd.read_csv('ad_costs.csv')


# **Обзор данных**

# In[3]:


#напишем функцию для обзора данных
def overlook(data):
    display(data.head(),
    data.describe(),
    data.info(),
    data.isna().sum())
    print('Кол-во дубликатов:', data.duplicated().sum())


# In[4]:


overlook(game_actions)


# **Вывод:**
# * в столбце building_type 7683 пропущенных значений, что возможно, игроки могли ничего не строить, трогать это не будем
# * в столбце одно уникальное значение и это satellite_orbital_assembly, остальные 133774 значения это пропуски
# * в датасете всего один дубликат, от него можно будет избавиться
# * в event_datetime изменить тип данных на правильный

# In[5]:


overlook(user_source)


# **Вывод:**
# * пропусков и дубликатов нет
# * количество уникальных пользователей совпадает с подсчётом всех пользователей

# In[6]:


overlook(ad_costs)


# In[7]:


print(ad_costs['source'].unique())


# **Вывод:**
# * всего 4 разных источника рекламы
# * пропусков и дубликатов нет
# * в столбце day изменить тип данных на datetime

# **Предобработка данных**

# In[8]:


#приведём столбцы в датафреймах к нужному типу данных 
game_actions['event_datetime'] = pd.to_datetime(game_actions['event_datetime'])
display(game_actions.head())
game_actions.info()


# In[9]:


ad_costs['day'] = pd.to_datetime(ad_costs['day'])
display(ad_costs.head())
ad_costs.info()


# In[10]:


#удалим один дубликат
game_actions = game_actions.drop_duplicates().reset_index(drop=True)
game_actions.duplicated().sum()


# In[11]:


#добавим нужные для анализа столбцы
game_actions['date'] = game_actions['event_datetime'].dt.date


# # 2. Исследовательский анализ данных

# **2.1 Проанализировать распределение количества пользователей из каждого источника**

# In[12]:


#объеденим таблицы game_actions и user_source
game_actions = game_actions.merge(user_source, how='left', on='user_id')
game_actions.head()


# In[13]:


per_source = game_actions.groupby('source').agg(user_count= ('user_id', 'nunique')).sort_values(by='user_count', ascending=False)
per_source


# In[14]:


fig = px.bar(per_source,
            x='user_count',
            )
fig.update_layout(title='Распределение количества пользователей из каждого источника',
                 xaxis_title='кол-во пользователей',
                 yaxis_title='источник')
fig.show()


# **Вывод**
# * Самое большое количество пользователей пришло из yandex_direct(4817), потом идёт instagram_new_adverts(3347)

# **2.2 Проанализировать подходы к игровым стратегиям пользователей**

# В игре существует два вида выполнения уровня:
# * Путь исследования(разработать satellite_orbital_assembly)
# * Победа над врагом

# In[15]:


#посчитаем количество игроков, прошедших первый уровень
finished_level = game_actions[game_actions['event'] == 'finished_stage_1']['user_id'].count()
print('Всего игроков завершивших уровень:', finished_level)


# In[16]:


#посчитаем игроков, прошедших уровень путём исследования
sience_victory = game_actions[game_actions['project_type'] == 'satellite_orbital_assembly']['user_id'].count()
print('Всего игроков завершивших уровень научной победой:', sience_victory)


# In[17]:


#посчитаем игроков, прошедших уровень путём победы над врагом
fighter_victory = finished_level - sience_victory
print('Всего игроков завершивших уровень победой над врагом:', fighter_victory)


# In[18]:


finished = pd.Series([sience_victory, fighter_victory], index=['Научная победа', 'Воинственная победа']).reset_index()
finished


# In[19]:


fig = px.pie(finished,
             values = 0, 
             names = ('Научная победа', 'Воинственная победа'),
             title='Количество игроков с разными стратегиями, прошедших первый уровень',
             width=700, 
             height=500,
             color_discrete_sequence=[px.colors.qualitative.Pastel[0],
                                     px.colors.qualitative.Pastel[1]])
fig.update_traces(textposition='inside', textinfo='percent+label')
fig.show()


# **Вывод**
# * Большинство игроков(67.9%) завершают уровень победой над врагом

# **Проанализировать кол-во построек, которое построили игроки из каждой из групп**

# In[20]:


#построим таблицу с подробной информацией о том что строил каждый пользователей и из какого источника он пришёл
game_actions_new = game_actions.groupby(['user_id', 'event'])['user_id'].count().unstack(level=1).reset_index()
game_actions_new.head()


# In[21]:


game_actions_new = game_actions_new.drop(columns = ['finished_stage_1', 'project'])


# In[22]:


building_count = game_actions.groupby(['user_id', 'building_type'])['user_id'].count().unstack(level=1).reset_index()
game_actions_new = game_actions_new.merge(building_count, how='left', on='user_id')
game_actions_new.head()


# In[23]:


game_actions_new = game_actions_new.merge(user_source, how='left', on='user_id')
game_actions_new.head()


# In[24]:


source_building = game_actions_new.groupby('source').agg(total_buildings = ('building', 'sum'),
                                                       spaceport = ('spaceport', 'sum'),
                                                       assembly_shop = ('assembly_shop', 'sum'),
                                                       research_center = ('research_center', 'sum')).reset_index()
source_building.sort_values('total_buildings', ascending=False, inplace=True)
source_building


# In[25]:


source_building_new = source_building.melt(id_vars=['source'], value_vars=['total_buildings', 'spaceport', 'assembly_shop', 'research_center'],
                 var_name='buildings', value_name='count')
source_building_new.head()


# In[26]:


sns.set_style('dark')
plt.figure(figsize=(15, 5))
sns.barplot(x='source', y='count', data=source_building_new, hue='buildings', palette='crest')
plt.title('Количество построек по каждому источнику')
plt.xlabel('Источники')
plt.ylabel('Количество построек')
plt.legend(loc='upper right', fontsize=10)
plt.grid()
plt.show()


# **Вывод**
# * Больше всего было построено spaceport, далее assembly_shop и на последнем месте research_center
# * Каких то существенных разниц между источниками не замечается

# **2.3 Проанализировать метрики**

# In[27]:


#построим таблицу с минимальной и максимальной датой активности у каждого пользователя и с числом затрат на каждый день
dynamics = game_actions.groupby('user_id').agg({'event_datetime':['min','max'],                                                        
                                                        'source': 'min'}).reset_index()
dynamics.columns = ['user_id', 'min_date','max_date','source']
dynamics.head()


# In[28]:


dynamics['sale_date'] = pd.to_datetime(dynamics['min_date']).dt.date
dynamics['sale_date'] = pd.to_datetime(dynamics['sale_date'])
dynamics.head()


# In[29]:


#видимо запуск игры произошёл через день после начала рекламной кампании чтобы совместить таблицы добавим один день к дате начала рекламной акции
ad_costs['sale_date'] = ad_costs['day'] + pd.DateOffset(days=1)


# In[30]:


metrics = dynamics.merge(ad_costs, how='outer', on=['source','sale_date'])
metrics.head()


# In[31]:


metrics['cost'].unique()


# In[32]:


#построим таблицу для дальнейшей визуализации динамики привлечения игроков
user_dynamic = metrics.pivot_table(index='sale_date', 
                                   columns='source',  
                                   values='user_id',  
                                   aggfunc='nunique')

user_dynamic


# In[33]:


user_dynamic.plot(figsize=(16, 6), grid=True)
plt.title('Динамика привлечения игроков')
plt.xlabel('Дата')
plt.ylabel('Количество игроков')
plt.show()


# Наибольшее количество пользователей привлёк yandex_direct, на втором месте instagram. Ближе к окончанию рекламной кампании показатели источников стали примерно равны

# In[34]:


#построим таблицу для визуализации расходов рекламной кампании
ad_dynamics = ad_costs.pivot_table(index='sale_date',
                                   columns='source', 
                                   values='cost',  
                                   aggfunc='sum')

ad_dynamics


# In[35]:


ad_cost_count = (ad_costs
             .groupby(['sale_date','source'], as_index=False)
             .agg(count=('cost','sum'))
             .sort_values(by='count', ascending=False)
            )
ad_cost_count['sale_date'] = ad_cost_count['sale_date'].dt.date
ad_cost_count.head()


# In[36]:


sns.set_style('dark')
plt.figure(figsize=(16, 6))
sns.barplot(x='sale_date', y='count', data=ad_cost_count, hue='source', palette='crest')
plt.title('Затраты на пользователя по датам')
plt.xlabel('Дата')
plt.ylabel('Затраты')
plt.legend(loc='upper right', fontsize=10)
plt.grid()
plt.show()


# Самые высокие затраты на рекламу уходят на yandex_direct, который показывает самые высокие показатели по привлечению пользователей. Почти столько же трат уходит на facebook, который в свою очередь привлекает пользователей на уровне youtube На youtube уходит меньше всего затрат, в два раза меньше чем на другие источники.

# In[37]:


#посчитаем cac
cac = ad_dynamics/user_dynamic
cac = cac.mean()
cac = cac.to_frame()
cac.columns = [ 'cac']
cac


# **Выводы:**
# * Дороже всего нам обходятся пользователи из facebook, дешевле всего пользователи с youtube
# * Динамика по привлечению пользователей у facebook и youtube практически равна

# # 3. Проверка статистических гипотез

# **3.1 Проверить гипотезу: время завершения уровня различается в зависимости способа прохождения: нулевая - через реализацию проекта, альтернативная - через победу над первым игроком.**

# In[38]:


#создадим таблицу с минимальным и максимальным временем событий, а также добавим разницу между ними
min_event = game_actions.groupby(['user_id']).agg(first_event_datetime = ('event_datetime', 'min')).reset_index()
max_event = game_actions.groupby(['user_id']).agg(last_event_datetime = ('event_datetime', 'max')).reset_index()


# In[39]:


date_event = pd.merge(min_event, max_event, how = 'left', on = 'user_id')
date_event['hours'] = (date_event['last_event_datetime'] - date_event['first_event_datetime']).astype('timedelta64[h]')
date_event.head()


# In[40]:


#создадим списки, из которых будем брать id людей с разными стратегиями
science_id = game_actions.query('project_type == "satellite_orbital_assembly"')['user_id'].to_list()
warrior_id = game_actions.query('(event == "finished_stage_1") and (user_id != @science_id)')['user_id'].to_list()


# In[41]:


science_time = date_event.query('user_id == @science_id')
warriors_time = date_event.query('user_id == @warrior_id')


# Сформулируем гипотезы
# 
# H0 будет звучать так:
# 
# Среднее время прохождения уровня между игроками не различается в зависимости от стратегии
# 
# H1 будет звучать:
# 
# Среднее время прохождения уровня между игроками различается в зависимости от стратегии

# In[42]:


alpha = 0.05

results = st.ttest_ind(warriors_time['hours'], science_time['hours'])

print('p-значение: ', results.pvalue)
if (results.pvalue < alpha):
    print("Отвергаем нулевую гипотезу, время прохождения уровня между игроками различается")
else:
    print("Не получилось отвергнуть нулевую гипотезу, время прохождения уровня между игроками не различается")
print('в среднем: {:.0f} часов при победе над врагом и {:.0f} часа при научной победе'.format(date_event.query('user_id == @warrior_id')['hours'].mean(), date_event.query('user_id == @science_id')['hours'].mean()))


# **3.2 Проверить гипотезу: влияет ли источник на количество совершённых событий пользователем: нулевая - количестов действий зависит от источника альтернативная - количество действий не зависит от источника**

# Сформулируем гипотезы
# 
# H0 будет звучать так:
# 
# Среднее количество событий не отличается в зависимости от источника трафика
# 
# H1 будет звучать:
# 
# Среднее количество событий отличается в зависимости от источника трафика

# In[43]:


game_actions.head()


# In[44]:


yandex_id = game_actions.query('source == "yandex_direct"')['user_id'].to_list()
facebook_id = game_actions.query('source == "facebook_ads"')['user_id'].to_list()
instagram_id = game_actions.query('source == "instagram_new_adverts"')['user_id'].to_list()
youtube_id = game_actions.query('source == "youtube_channel_reklama"')['user_id'].to_list()


# In[45]:


count_events = game_actions.groupby('user_id')['event'].count().reset_index()
count_events.head()


# In[46]:


yandex_events = count_events.query('user_id == @yandex_id')
facebook_events = count_events.query('user_id == @facebook_id')
instagram_events = count_events.query('user_id == @instagram_id')
youtube_events = count_events.query('user_id == @youtube_id')
yandex_events.head()


# In[47]:


def test(a, b, c, d):
    results = st.ttest_ind(a['event'], b['event'])
    print('p-значение: ', results.pvalue)
    
    if (results.pvalue < alpha):
        print("Отвергаем нулевую гипотезу, количество событий отличается в зависимости от источника трафика")
        print(round(count_events.query('user_id == @c')['event'].mean(), 1),               round(count_events.query('user_id == @d')['event'].mean(), 1))
    else:
        print("Не получилось отвергнуть нулевую гипотезу, количество событий не отличается в зависимости от источника трафика")


# In[48]:


test(yandex_events, facebook_events, yandex_id, facebook_id)


# In[49]:


test(yandex_events, instagram_events, yandex_id, instagram_id)


# In[50]:


test(yandex_events, youtube_events, yandex_id, youtube_id)


# In[51]:


test(facebook_events, instagram_events, facebook_id, instagram_id)


# In[52]:


test(facebook_events, youtube_events, facebook_id, youtube_id)


# In[53]:


test(instagram_events, youtube_events, instagram_id, youtube_id)


# **Выводы по гипотезам:**
# * В среднем игроки проводят в игре 266 часов при победе над врагом и 323 часа при научной победе, при этом игроки предпочитают играть стратегией победы над врагом
# * Игроки пришедшие из facebook совершают больше всгео действий, но я бы не скзала что разница значительная с учётом больших затрат на рекламу в facebook и маленький приток игроков

# # 4. Выводы и рекомендации

# **Выводы:**
# * Самое большое количество пользователей пришло из yandex_direct(4817), потом идёт instagram_new_adverts(3347). Большинство игроков(67.9%) завершают уровень победой над врагом. Больше всего было построено spaceport, далее assembly_shop и на последнем месте research_center. Каких то существенных разниц между источниками не замечается. Наибольшее количество пользователей привлёк yandex_direct, на втором месте instagram. Ближе к окончанию рекламной кампании показатели источников стали примерно равны. Самые высокие затраты на рекламу уходят на yandex_direct, который показывает самые высокие показатели по привлечению пользователей. Почти столько же трат уходит на facebook, который в свою очередь привлекает пользователей на уровне youtube На youtube уходит меньше всего затрат, в два раза меньше чем на другие источники. Дороже всего нам обходятся пользователи из facebook, дешевле всего пользователи с youtube. Показатели по привлечению у facebook и youtube практически равны. В среднем игроки проводят в игре 266 часов при победе над врагом и 323 часа при научной победе, при этом игроки предпочитают играть стратегией победы над врагом. Игроки пришедшие из facebook совершают больше всгео действий, но я бы не скзала что разница значительная с учётом больших затрат на рекламу в facebook и маленький приток игроков

# **Рекомендации:**
# * Больше прибыли принесут пользователи, предпочитающие стратегию исследования, а не сражения с врагом. Стратегию победы над врагом игроки выбирают в два раза чаще чем стратегию исследования. Возможно, стоит сбалансировать игровые стратегии по времени игры, что-то изменить, чтобы сделать строительство более привлекательным для игроков.
# * Из исследования можно выделить два более перспективных канала привлечения: yandex и youtube. Рекомендую обратить на них больше внимания и уделить на них больше рекламного бюджета.

# In[ ]:




