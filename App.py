import streamlit as st
import pickle
import requests
from huggingface_hub import hf_hub_download

from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

st.set_page_config(
    page_title="Movie Recommender System",
    page_icon="🎬",
    layout="wide"
)

try:
    API_KEY = st.secrets["TMDB_API_KEY"]

    # Hugging Face configuration
    HF_REPO_ID = "thebjax/movie-recommender-data"
    HF_REPO_TYPE = "model"

except (KeyError, FileNotFoundError):
    st.error(
        "TMDB API key not found. "
        "Please add TMDB_API_KEY to "
        ".streamlit/secrets.toml."
    )
    st.stop()

session = requests.Session()

retry_strategy = Retry(
    total=3,
    connect=3,
    read=2,
    backoff_factor=1,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["GET"]
)

adapter = HTTPAdapter(max_retries=retry_strategy)

session.mount("https://", adapter)
session.mount("http://", adapter)


@st.cache_resource
def load_data():
    hf_token = st.secrets["HF_TOKEN"]
    movies_path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename="movies.pkl",
        repo_type=HF_REPO_TYPE,
        token=hf_token
    )
    similarity_path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename="similarity.pkl",
        repo_type=HF_REPO_TYPE,
        token=hf_token
    )
    with open(movies_path, "rb") as file:
        movies_data = pickle.load(file)
    with open(similarity_path, "rb") as file:
        similarity_data = pickle.load(file)
    return movies_data, similarity_data


try:
    movies, similarity = load_data()

except FileNotFoundError:
    st.error(
        "Could not find movies.pkl or similarity.pkl. "
        "Place both files in the same folder as App.py."
    )
    st.stop()

except (pickle.UnpicklingError, EOFError) as error:
    st.error(f"Could not load your pickle files: {error}")
    st.stop()



if "title" not in movies.columns:
    st.error(
        "Your movies.pkl file must contain a 'title' column."
    )
    st.stop()

if len(movies) != len(similarity):
    st.error(
        "The number of movies does not match the similarity "
        "matrix. Please check your pickle files."
    )
    st.stop()

if movies["title"].dropna().empty:
    st.error("No movie titles were found in movies.pkl.")
    st.stop()


@st.cache_data(ttl=3600)
def fetch_movie_details(movie_name):

    url = "https://api.themoviedb.org/3/search/movie"

    params = {
        "api_key": API_KEY,
        "query": movie_name,
        "include_adult": False
    }

    try:

        response = session.get(
            url,
            params=params,
            timeout=20
        )

        response.raise_for_status()

        results = response.json().get("results", [])

        print(f"\nSearching TMDB for: {movie_name}")
        print(f"HTTP Status: {response.status_code}")
        print(f"Results found: {len(results)}")

        # No search results
        if not results:
            print(f"No TMDB results found for: {movie_name}")
            return None

        # Find exact title matches
        exact_matches = [
            movie for movie in results
            if movie.get("title", "").strip().casefold()
            == movie_name.strip().casefold()
        ]

        # Prefer exact matches over unrelated movies
        candidates = exact_matches if exact_matches else results

        # Prefer a matching result with a poster
        selected_movie = next(
            (
                movie for movie in candidates
                if movie.get("poster_path")
            ),
            candidates[0]
        )

        poster_path = selected_movie.get("poster_path")

        poster_url = (
            f"https://image.tmdb.org/t/p/w500{poster_path}"
            if poster_path
            else None
        )

        # Release year
        release_date = selected_movie.get("release_date", "")

        release_year = (
            release_date[:4]
            if release_date
            else "Unknown"
        )

        # Movie overview
        overview = (
            selected_movie.get("overview")
            or "No description available."
        )

        # Movie rating
        rating = selected_movie.get("vote_average", 0)

        print(f"Selected title: {selected_movie.get('title')}")
        print(f"Poster available: {poster_url is not None}")

        return {
            "poster": poster_url,
            "year": release_year,
            "overview": overview,
            "rating": rating
        }

    except requests.RequestException as error:

        print(
            f"TMDB request failed for {movie_name}: {error}"
        )

        return None

    except (ValueError, TypeError) as error:

        print(
            f"Invalid TMDB response for {movie_name}: {error}"
        )

        return None


def recommend(movie_name):

    matching_movies = movies[
        movies["title"] == movie_name
    ]

    if matching_movies.empty:
        return []

    # Find the selected movie's position
    movie_index = movies.index.get_loc(
        matching_movies.index[0]
    )

    # Get similarity scores
    distances = similarity[movie_index]

    # Sort movies by similarity score
    sorted_movies = sorted(
        enumerate(distances),
        key=lambda item: item[1],
        reverse=True
    )

    recommendations = []

    for index, score in sorted_movies:

        # Do not recommend the selected movie itself
        if index == movie_index:
            continue

        title = movies.iloc[index]["title"]

        if not isinstance(title, str):
            continue

        recommendations.append(title)

        # Return five recommendations
        if len(recommendations) == 5:
            break

    return recommendations



st.title("🎬 Movie Recommender System")

st.write(
    "Discover movies similar to your favorites "
    "using content-based filtering."
)

st.divider()



movie_titles = (
    movies["title"]
    .dropna()
    .drop_duplicates()
    .sort_values()
    .tolist()
)

selected_movie = st.selectbox(
    "🎥 Select a movie you like:",
    movie_titles
)

recommend_button = st.button(
    "🍿 Recommend Movies",
    type="primary",
    use_container_width=True
)


if recommend_button:

    with st.spinner("Finding movies you might enjoy..."):

        recommendations = recommend(selected_movie)

    if not recommendations:

        st.warning("No recommendations found.")

    else:

        st.subheader(
            f"Movies similar to {selected_movie}"
        )

        columns = st.columns(5)

        for column, movie_name in zip(
            columns,
            recommendations
        ):

            with column:

                # Always display the movie title
                st.markdown(f"### {movie_name}")

                # Fetch movie details
                details = fetch_movie_details(movie_name)

                if details is None:

                    st.warning(
                        "Movie details could not be retrieved."
                    )

                    st.caption(
                        "Please try again later."
                    )

                    continue

                # Display poster
                if details["poster"]:

                    st.image(
                        details["poster"],
                        width="stretch"
                    )

                else:

                    st.info(
                        "🖼️ Poster unavailable for this movie."
                    )

                # Display release year
                st.caption(
                    f"📅 Release year: {details['year']}"
                )

                # Display rating
                rating = details["rating"]

                if rating:
                    st.caption(
                        f"⭐ Rating: {rating:.1f}/10"
                    )
                else:
                    st.caption("⭐ Rating unavailable")

                # Display movie overview
                with st.expander("Movie details"):
                    st.write(details["overview"])



st.divider()

st.caption(
    "Movie recommendations are generated using your "
    "similarity model. Movie information and posters "
    "are retrieved from TMDB."
)

st.caption(
    "This product uses the TMDB API but is not endorsed "
    "or certified by TMDB."
)