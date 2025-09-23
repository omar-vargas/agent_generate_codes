import streamlit as st
import requests
import json
import pandas as pd
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Dict, Any
import base64

# Initialize session state
if "search_results" not in st.session_state:
    st.session_state.search_results = []
if "search_facets" not in st.session_state:
    st.session_state.search_facets = {}
if "selected_filters" not in st.session_state:
    st.session_state.selected_filters = {
        "categories": [],
        "ontologies": [],
        "sessions": [],
        "users": [],
        "date_from": None,
        "date_to": None
    }

# Function to get auth headers
def get_auth_headers():
    if st.session_state.get("access_token"):
        return {"Authorization": f"Bearer {st.session_state.access_token}"}
    return {}

# Function to perform search
def perform_search(search_params):
    try:
        headers = get_auth_headers()
        response = requests.post("http://localhost:8000/search/", json=search_params, headers=headers)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Search failed: {response.status_code}")
            return []
    except Exception as e:
        st.error(f"Search error: {e}")
        return []

# Function to get search facets
def get_search_facets():
    try:
        headers = get_auth_headers()
        response = requests.get("http://localhost:8000/search/facets/", headers=headers)
        if response.status_code == 200:
            return response.json()
        return {}
    except Exception as e:
        st.error(f"Error loading facets: {e}")
        return {}

# Function to find similar codes
def find_similar_codes(code_ids, limit=10, threshold=0.7):
    try:
        headers = get_auth_headers()
        response = requests.post("http://localhost:8000/search/similar-codes/",
                               json={"code_ids": code_ids, "limit": limit, "threshold": threshold},
                               headers=headers)
        if response.status_code == 200:
            return response.json()
        return {"similar_codes": []}
    except Exception as e:
        st.error(f"Error finding similar codes: {e}")
        return {"similar_codes": []}

# Function to search hierarchical relationships
def search_hierarchical(code_id, direction="both", max_depth=3):
    try:
        headers = get_auth_headers()
        response = requests.post("http://localhost:8000/search/hierarchical/",
                               json={"code_id": code_id, "direction": direction, "max_depth": max_depth},
                               headers=headers)
        if response.status_code == 200:
            return response.json()
        return {}
    except Exception as e:
        st.error(f"Error searching hierarchy: {e}")
        return {}

# Function to export search results
def export_search_results(results, format_type="csv"):
    if format_type == "csv":
        df = pd.DataFrame([{
            "Type": r["type"],
            "Title": r["title"],
            "Category": r["category"],
            "Ontology": r["ontology"],
            "Session": r["session_id"],
            "Created": r["created_at"],
            "Similarity": r["similarity_score"]
        } for r in results])
        return df.to_csv(index=False)
    elif format_type == "json":
        return json.dumps(results, indent=2, ensure_ascii=False)

# Main search interface
def main():
    # Only set page config if not already set (to avoid conflicts when imported)
    if "page_config_set" not in st.session_state:
        st.set_page_config(page_title="Advanced Search - Qualitative Analysis", page_icon="🔍", layout="wide")
        st.session_state.page_config_set = True

    st.title("🔍 Advanced Search & Filtering")

    # Back button
    col1, col2 = st.columns([1, 11])
    with col1:
        if st.button("← Back"):
            st.session_state.current_page = "main_workflow"
            st.rerun()

    # Check authentication
    if not st.session_state.get("logged_in"):
        st.warning("Please log in to access the search interface.")
        return

    # Sidebar for filters
    with st.sidebar:
        st.header("🎛️ Search Filters")

        # Load facets if not loaded
        if not st.session_state.search_facets:
            st.session_state.search_facets = get_search_facets()

        # Search type
        search_type = st.selectbox(
            "Search Scope",
            ["all", "codes", "documents", "categories", "ontologies"],
            help="What type of content to search"
        )

        # Categories filter
        if st.session_state.search_facets.get("categories"):
            st.subheader("📂 Categories")
            selected_categories = []
            for cat in st.session_state.search_facets["categories"]:
                if st.checkbox(f"{cat['name']}", key=f"cat_{cat['id']}"):
                    selected_categories.append(cat['id'])
            st.session_state.selected_filters["categories"] = selected_categories

        # Ontologies filter
        if st.session_state.search_facets.get("ontologies"):
            st.subheader("🧠 Ontologies")
            selected_ontologies = []
            for ont in st.session_state.search_facets["ontologies"]:
                if st.checkbox(f"{ont['name']}", key=f"ont_{ont['id']}"):
                    selected_ontologies.append(ont['id'])
            st.session_state.selected_filters["ontologies"] = selected_ontologies

        # Sessions filter
        if st.session_state.search_facets.get("sessions"):
            st.subheader("📋 Sessions")
            selected_sessions = []
            for sess in st.session_state.search_facets["sessions"]:
                if st.checkbox(f"{sess['name']}", key=f"sess_{sess['id']}"):
                    selected_sessions.append(sess['id'])
            st.session_state.selected_filters["sessions"] = selected_sessions

        # Date range filter
        st.subheader("📅 Date Range")
        date_from = st.date_input("From date", value=None, key="date_from")
        date_to = st.date_input("To date", value=None, key="date_to")

        if date_from:
            st.session_state.selected_filters["date_from"] = date_from.isoformat() + "Z"
        else:
            st.session_state.selected_filters["date_from"] = None

        if date_to:
            st.session_state.selected_filters["date_to"] = date_to.isoformat() + "Z"
        else:
            st.session_state.selected_filters["date_to"] = None

        # Advanced options
        st.subheader("⚙️ Advanced Options")
        semantic_search = st.checkbox("🔍 Semantic Search", help="Use AI embeddings for semantic similarity")
        similarity_threshold = st.slider("Similarity Threshold", 0.1, 1.0, 0.7, 0.1,
                                       help="Minimum similarity score for semantic search")
        include_hierarchy = st.checkbox("🌳 Include Hierarchy", help="Show hierarchical relationships in results")

    # Main search area
    col1, col2 = st.columns([3, 1])

    with col1:
        st.subheader("🔎 Search Query")
        query = st.text_input(
            "Enter your search terms",
            placeholder="Search codes, documents, categories, ontologies...",
            help="Enter keywords to search across your qualitative data"
        )

        # Search button
        if st.button("🔍 Search", type="primary", use_container_width=True):
            if not query and not semantic_search:
                st.warning("Please enter a search query or enable semantic search.")
            else:
                with st.spinner("Searching..."):
                    search_params = {
                        "query": query,
                        "search_type": search_type,
                        "category_ids": st.session_state.selected_filters["categories"],
                        "ontology_ids": st.session_state.selected_filters["ontologies"],
                        "session_ids": st.session_state.selected_filters["sessions"],
                        "date_from": st.session_state.selected_filters["date_from"],
                        "date_to": st.session_state.selected_filters["date_to"],
                        "semantic_search": semantic_search,
                        "similarity_threshold": similarity_threshold,
                        "include_hierarchy": include_hierarchy,
                        "limit": 100
                    }

                    results = perform_search(search_params)
                    st.session_state.search_results = results

                    if results:
                        st.success(f"Found {len(results)} results")
                    else:
                        st.info("No results found. Try adjusting your search criteria.")

    with col2:
        # Quick actions
        st.subheader("⚡ Quick Actions")

        if st.button("🔄 Clear Filters", use_container_width=True):
            st.session_state.selected_filters = {
                "categories": [],
                "ontologies": [],
                "sessions": [],
                "users": [],
                "date_from": None,
                "date_to": None
            }
            st.rerun()

        if st.button("📊 View Statistics", use_container_width=True):
            # Show search statistics
            if st.session_state.search_results:
                results_df = pd.DataFrame(st.session_state.search_results)

                # Type distribution
                type_counts = results_df['type'].value_counts()
                st.write("**Results by Type:**")
                for type_name, count in type_counts.items():
                    st.write(f"- {type_name.title()}: {count}")

                # Average similarity
                if 'similarity_score' in results_df.columns:
                    avg_similarity = results_df['similarity_score'].mean()
                    st.write(".2f")

    # Results display
    if st.session_state.search_results:
        st.header("📋 Search Results")

        # Results summary
        results_df = pd.DataFrame(st.session_state.search_results)

        # Filters and sorting
        col1, col2, col3 = st.columns(3)

        with col1:
            sort_by = st.selectbox("Sort by", ["relevance", "date", "type"], index=0)

        with col2:
            result_type_filter = st.multiselect(
                "Filter by type",
                options=results_df['type'].unique(),
                default=results_df['type'].unique()
            )

        with col3:
            if 'similarity_score' in results_df.columns:
                min_similarity = st.slider("Min similarity", 0.0, 1.0, 0.0, 0.1)

        # Apply filters
        filtered_results = results_df[
            (results_df['type'].isin(result_type_filter)) &
            (results_df['similarity_score'] >= min_similarity)
        ]

        # Sort results
        if sort_by == "relevance":
            filtered_results = filtered_results.sort_values('similarity_score', ascending=False)
        elif sort_by == "date":
            filtered_results = filtered_results.sort_values('created_at', ascending=False)
        elif sort_by == "type":
            filtered_results = filtered_results.sort_values(['type', 'similarity_score'], ascending=[True, False])

        # Display results
        for idx, result in filtered_results.iterrows():
            with st.expander(f"{'🔹' if result['type'] == 'code' else '📄' if result['type'] == 'document' else '📂' if result['type'] == 'category' else '🧠'} {result['title']} ({result['type'].replace('_', ' ').title()})", expanded=False):

                # Metadata
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.write(f"**Type:** {result['type'].replace('_', ' ').title()}")
                with col2:
                    if result['similarity_score'] > 0:
                        st.write(".2f")
                with col3:
                    st.write(f"**Date:** {pd.to_datetime(result['created_at']).strftime('%Y-%m-%d')}")

                # Content
                if result['content']:
                    st.write("**Content:**")
                    st.text_area("", value=result['content'], height=100, disabled=True, key=f"content_{idx}")

                # Additional metadata
                if result['category']:
                    st.write(f"**Category:** {result['category']}")
                if result['ontology']:
                    st.write(f"**Ontology:** {result['ontology']}")
                if result['session_id']:
                    st.write(f"**Session:** {result['session_id'][:12]}...")

                # Hierarchy path
                if result.get('hierarchy_path') and len(result['hierarchy_path']) > 1:
                    st.write(f"**Hierarchy:** {' → '.join(result['hierarchy_path'])}")

                # Action buttons
                action_col1, action_col2, action_col3 = st.columns(3)

                with action_col1:
                    if result['type'] == 'code':
                        if st.button("🔗 Find Similar", key=f"similar_{result['id']}"):
                            similar = find_similar_codes([int(result['id'])])
                            if similar['similar_codes']:
                                st.subheader("Similar Codes:")
                                for sim in similar['similar_codes'][:5]:
                                    st.write(f"- {sim['name']} (similarity: {sim['similarity']:.2f})")

                with action_col2:
                    if result['type'] == 'code':
                        if st.button("🌳 Explore Hierarchy", key=f"hierarchy_{result['id']}"):
                            hierarchy = search_hierarchical(int(result['id']))
                            if hierarchy.get('ancestors') or hierarchy.get('descendants'):
                                st.subheader("Hierarchical Relationships:")

                                if hierarchy.get('ancestors'):
                                    st.write("**Ancestors:**")
                                    for anc in hierarchy['ancestors']:
                                        st.write(f"  {'  ' * anc['depth']}└─ {anc['name']}")

                                if hierarchy.get('descendants'):
                                    st.write("**Descendants:**")
                                    for desc in hierarchy['descendants']:
                                        st.write(f"  {'  ' * desc['depth']}└─ {desc['name']}")

                with action_col3:
                    if st.button("📋 Copy Details", key=f"copy_{result['id']}"):
                        details = f"Type: {result['type']}\nTitle: {result['title']}\nContent: {result['content']}"
                        st.code(details)

        # Export options
        st.header("💾 Export Results")

        col1, col2, col3 = st.columns(3)

        with col1:
            csv_data = export_search_results(st.session_state.search_results, "csv")
            st.download_button(
                "⬇️ Download CSV",
                data=csv_data,
                file_name="search_results.csv",
                mime="text/csv",
                use_container_width=True
            )

        with col2:
            json_data = export_search_results(st.session_state.search_results, "json")
            st.download_button(
                "⬇️ Download JSON",
                data=json_data,
                file_name="search_results.json",
                mime="application/json",
                use_container_width=True
            )

        with col3:
            # Visualization option
            if st.button("📊 Visualize Results", use_container_width=True):
                # Create visualization
                fig = px.bar(
                    results_df.groupby('type').size().reset_index(name='count'),
                    x='type',
                    y='count',
                    title="Results by Type",
                    labels={'type': 'Content Type', 'count': 'Number of Results'}
                )
                st.plotly_chart(fig, use_container_width=True)

    else:
        # Welcome message and tips
        st.info("👋 Welcome to Advanced Search! Enter a query above or use the filters in the sidebar to start searching your qualitative data.")

        st.subheader("💡 Search Tips")
        st.markdown("""
        - **Text Search**: Enter keywords to find exact matches in codes, documents, categories, and ontologies
        - **Semantic Search**: Enable AI-powered semantic search to find conceptually similar content
        - **Filters**: Use the sidebar to narrow down results by categories, ontologies, sessions, and date ranges
        - **Hierarchy**: Include hierarchical relationships to see parent/child relationships in results
        - **Similarity**: Adjust the similarity threshold for semantic search results
        """)

if __name__ == "__main__":
    main()