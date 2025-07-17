# 🚚 Logistics Optimization using CrewAI + Streamlit

This is a Streamlit web application that uses AI agents (powered by Google's Gemini via LangChain and CrewAI) to analyze logistics operations and generate smart optimization strategies.

---

## 📌 Features

- 🌐 Gemini-powered AI agents using `langchain-google-genai`
- 🧠 CrewAI Agents: Logistics Analyst & Optimization Strategist
- 🖥️ Interactive Streamlit UI for entering product data
- 📊 AI-generated summary and actionable logistics strategy

---

## 🛠️ Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt


Sample inputs

TV, Laptops, Headphones

Output


🔍 Final Optimization Strategy
Optimization Strategy to Reduce Delivery Time and Improve Inventory Management

This strategy addresses the inefficiencies identified in delivery routes and inventory turnover for TVs, Laptops, and Headphones, based on the Logistics Analyst's findings. The strategy focuses on actionable steps to improve logistics efficiency and reduce costs.

I. Delivery Route Optimization:

A. Route Optimization Software Implementation:

Action Point 1: Implement a route optimization software solution (e.g., [Specify a suitable software, if known, otherwise leave blank]). This software will analyze historical delivery data (distance, time, stops, cost) to identify shorter, more efficient routes for all products. The software should consider factors like traffic patterns, delivery windows, and vehicle capacity.

Action Point 2: Prioritize optimization for Routes A (TVs - Northwest), B (Laptops - South), and C (Headphones - East), as these routes exhibit the most significant inefficiencies (high costs, long delivery times, inconsistent delivery times).

Action Point 3: Regularly monitor and re-optimize routes based on updated data and changing conditions (e.g., traffic patterns, road closures). The frequency of re-optimization should be determined based on the volatility of the data and the impact of changes on delivery efficiency.

B. Carrier Negotiation:

Action Point 4: Negotiate better rates with carriers, leveraging the data on route inefficiencies and cost analysis. Focus on routes with high costs relative to distance and number of stops. Explore options like consolidating shipments or using alternative carriers for specific routes.
II. Inventory Management Optimization:

A. Demand Forecasting Improvement:

Action Point 5: Implement or refine demand forecasting models using advanced techniques (e.g., time series analysis, machine learning). The models should incorporate historical sales data, seasonality, promotional activities, and external factors (e.g., economic conditions).

Action Point 6: Focus on improving demand forecasting accuracy for slow-moving items like the 55-inch Curved TV (Model X), 17-inch Gaming Laptop (Model Y), and Wireless Noise-Cancelling Headphones (Model Z). This might involve adjusting pricing, marketing strategies, or discontinuing underperforming products.

B. Inventory Management System Enhancement:

Action Point 7: Implement or upgrade the inventory management system to provide real-time visibility into inventory levels, sell-through rates, and lead times. The system should integrate with the route optimization software to ensure accurate stock allocation and efficient delivery scheduling.

Action Point 8: Explore implementing Just-in-Time (JIT) inventory management for slow-moving items. This will reduce holding costs by minimizing excess inventory while ensuring sufficient stock levels to meet demand.

C. Sales Data Analysis and Product Lifecycle Management:

Action Point 9: Conduct regular sales data analysis to identify slow-moving products early on. This will allow for proactive measures like price adjustments, promotional campaigns, or product discontinuation before significant inventory buildup occurs.

Action Point 10: Implement a product lifecycle management (PLM) system to track product performance, forecast demand, and manage inventory throughout the product's lifecycle.

III. Correlation Analysis and Continuous Improvement:

Action Point 11: Conduct a more in-depth correlation analysis to confirm the relationship between delivery route inefficiencies and slow-moving inventory. This will help identify specific areas where improvements in delivery times can directly impact inventory levels and reduce holding costs.

Action Point 12: Establish a continuous improvement process to regularly monitor key performance indicators (KPIs) such as delivery times, costs, inventory turnover rates, and customer satisfaction. Use this data to identify new areas for optimization and refine the strategy over time.

This comprehensive strategy, combining route optimization, improved demand forecasting, and enhanced inventory management, will significantly reduce delivery times, improve inventory efficiency, and ultimately reduce overall logistics costs. Regular monitoring and adjustments based on performance data are crucial for the long-term success of this optimization strategy.