from graphviz import Digraph

# สร้าง Digraph
dot = Digraph(comment='Hybrid Food Recommendation System', format='png')

# Start
dot.node('A', 'Start', shape='oval')

# Connect to DB
dot.node('B', 'Connect to SQL Server\nLoad tables: User_2, menus_2, Ratings\nLoad CSV: df_final2.csv, disease_prep.csv', shape='box')

# User Data Preparation
dot.node('C', 'User Data Preparation\n- Normalize columns\n- Calculate age, BMI, BMR, TDEE\n- Get latest mood', shape='box')

# Health constraints
dot.node('D', 'Health Constraints Filtering\n- Check user diseases\n- Filter avoid_nutrients & avoid_ingredients\n- Result = allowed_foods', shape='box')

# Content-based Recommendation
dot.node('E', 'Content-Based Recommendation\n- Filter diet_type & dislikes\n- Filter health constraints\n- Create user vector\n- Compute cosine similarity', shape='box')

# Collaborative Recommendation
dot.node('F', 'Collaborative Recommendation (SVD)\n- Build trainset from user_ratings\n- Train SVD\n- Predict scores\n- Filter allowed foods', shape='box')

# Hybrid Recommendation
dot.node('G', 'Hybrid Recommendation\n- Merge content & collaborative\n- Compute hybrid_score\n- Sort top N', shape='box')

# Output
dot.node('H', 'Output\n- User info\n- Top N recommended foods', shape='box')

# End
dot.node('I', 'End', shape='oval')

# วาดเส้น flow
dot.edges(['AB', 'BC', 'CD'])
dot.edge('D', 'E', label='Content-based')
dot.edge('D', 'F', label='Collaborative')
dot.edge('E', 'G')
dot.edge('F', 'G')
dot.edge('G', 'H')
dot.edge('H', 'I')

# บันทึกและ render เป็นไฟล์ png
dot.render('hybrid_recommendation_flowchart', view=True)
