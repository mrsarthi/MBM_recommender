def get_mood_cluster(genres_str, runtime_mins=0):
    g_lower = str(genres_str).lower()
    clusters = []
    if any(g in g_lower for g in ['science fiction', 'mystery', 'thriller', 'mind-bending']):
        clusters.append('Mind-Bending')
    if any(g in g_lower for g in ['noir', 'crime', 'drama', 'romance']):
        clusters.append('Late Night')
    if any(g in g_lower for g in ['action', 'adventure', 'war']):
        clusters.append('Popcorn & Adrenaline')
    if any(g in g_lower for g in ['comedy', 'animation', 'family']):
        clusters.append('Comfort')
    if runtime_mins > 0 and runtime_mins <= 105:
        clusters.append('Quick Watch')
    return clusters if clusters else ['General']
