import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Read and preprocess the data
df = pd.read_csv('The Cloud 100 - 2024.csv')

# Read the CSV data
df = pd.read_csv('The Cloud 100 - 2024.csv')

# Count companies by country
country_counts = df['Country'].value_counts()

# Process US data
us_df = df[df['Country'] == 'United States']
us_cities = us_df['HEADQUARTERS'].apply(lambda x: x.split(',')[0].strip())
sf_count = us_cities[us_cities == 'San Francisco'].count()
ny_count = us_cities[us_cities.isin(['New York', 'New York '])].count()
others_count = us_cities[~us_cities.isin(['San Francisco', 'New York', 'New York '])].count()

# Remove US from country_counts and sort other countries
us_count = country_counts['United States']
country_counts = country_counts.drop('United States').sort_values(ascending=True)

# Prepare data for plotting
countries = country_counts.index.tolist() + ['United States']
values = country_counts.values.tolist() + [us_count]

# Set up a color palette
colors = sns.color_palette("pastel", len(countries))
us_colors = sns.color_palette("Set2", 3)

# Create the plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10), gridspec_kw={'width_ratios': [3, 1]})

# Horizontal bar chart
bars = ax1.barh(countries, values, color=colors)
ax1.set_title('Distribution of Companies by Country', fontsize=16)
ax1.set_xlabel('Number of Companies', fontsize=12)
ax1.set_ylabel('Country', fontsize=12)

# Add value labels on the bars
for bar in bars:
    width = bar.get_width()
    ax1.text(width, bar.get_y() + bar.get_height()/2, f'{int(width)}',
             ha='left', va='center', fontsize=10, fontweight='bold',
             bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))

# US breakdown pie chart
us_data = [sf_count, ny_count, others_count]
us_labels = ['San Francisco', 'New York', 'Other US Cities']
wedges, texts, autotexts = ax2.pie(us_data, labels=us_labels, colors=us_colors, autopct='%1.1f%%', startangle=90)

# Enhance pie chart labels
for autotext in autotexts:
    autotext.set_fontsize(9)
    autotext.set_fontweight('bold')

ax2.set_title('US Companies Distribution', fontsize=16)

# Add total US count
ax2.text(0, -1.2, f'Total US Companies: {us_count}', ha='center', va='center', fontsize=12, fontweight='bold')

# Add a legend for the pie chart
ax2.legend(wedges, us_labels, title="US Cities", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))

# Adjust layout and display
plt.tight_layout()
plt.show()



def convert_to_float(value):
    if pd.isna(value) or value == '':
        return np.nan
    value = str(value).replace('$', '').strip()
    if value.endswith('B'):
        return float(value[:-1]) * 1e9
    elif value.endswith('M'):
        return float(value[:-1]) * 1e6
    else:
        try:
            return float(value)
        except ValueError:
            return np.nan

df['FUNDING'] = df['FUNDING'].apply(convert_to_float)
df['VALUATION'] = df['VALUATION'].apply(convert_to_float)
df['EMPLOYEES'] = pd.to_numeric(df['EMPLOYEES'], errors='coerce')

# Calculate funding efficiency
df['FUNDING_EFFICIENCY'] = df['VALUATION'] / df['FUNDING']

# Remove rows with missing data
df_clean = df.dropna(subset=['FUNDING', 'VALUATION', 'EMPLOYEES'])

# Create an interactive scatter plot
fig = px.scatter(df_clean,
                 x='FUNDING',
                 y='VALUATION',
                 size='EMPLOYEES',
                 color='Country',
                 hover_name='COMPANY',
                 hover_data=['CEO', 'WHAT IT DOES', 'FUNDING_EFFICIENCY'],
                 log_x=True,
                 log_y=True,
                 title='Funding vs Valuation for Cloud 100 Companies')

fig.update_layout(legend_title_text='Country')
fig.update_xaxes(title='Funding (log scale)')
fig.update_yaxes(title='Valuation (log scale)')

# Add a trend line
z = np.polyfit(np.log10(df_clean['FUNDING']), np.log10(df_clean['VALUATION']), 1)
p = np.poly1d(z)
fig.add_trace(go.Scatter(x=df_clean['FUNDING'],
                         y=10**p(np.log10(df_clean['FUNDING'])),
                         mode='lines',
                         name='Trend Line'))

# Highlight top 5 companies by valuation
top_5 = df_clean.nlargest(5, 'VALUATION')
fig.add_trace(go.Scatter(x=top_5['FUNDING'],
                         y=top_5['VALUATION'],
                         mode='markers+text',
                         marker=dict(size=20, symbol='star', color='gold'),
                         text=top_5['COMPANY'],
                         textposition='top center',
                         name='Top 5 by Valuation'))

# Create a subplot for additional charts
fig2 = make_subplots(rows=1, cols=2, subplot_titles=('Funding Distribution', 'Valuation Distribution'))

# Add funding distribution
fig2.add_trace(go.Box(y=df_clean['FUNDING'], name='Funding'), row=1, col=1)

# Add valuation distribution
fig2.add_trace(go.Box(y=df_clean['VALUATION'], name='Valuation'), row=1, col=2)

fig2.update_layout(title_text='Funding and Valuation Distributions')

# Show the plots
fig.show()
fig2.show()

# Calculate and print correlation
correlation = df_clean['FUNDING'].corr(df_clean['VALUATION'])
print(f"The correlation coefficient between funding and valuation is: {correlation:.2f}")

# Print top 5 companies by funding efficiency
print("\nTop 5 companies by funding efficiency:")
print(df_clean.nlargest(5, 'FUNDING_EFFICIENCY')[['COMPANY', 'FUNDING_EFFICIENCY']])