import zipfile
from io import BytesIO
from pathlib import Path

import numpy as np
import polars as pl
import requests
import scipy.sparse as sp

import entropy_balance_weighting as ebw


def main() -> None:
    """
    Run a real-world example of EBW reweighting.

    This takes the 1-year 2023 American Community Survey (ACS) housing data,
    and reweights the housing units so that the total housing counts by state
    match exactly the housing unit counts from the 2023 U.S. Census Population
    Estimates Program.

    There is not much of a reason to actually do this, as by construction the
    housing counts will almost match exactly with the base weights, but
    this process could be used to match smaller geographies when available, or
    information of number of housing units with residents of a certain age in them,
    if this information were externally available.
    """
    state_housing_estimates = download_state_housing_estimates()
    household_weights = download_and_process_acs_data()
    survey_weighting(household_weights, state_housing_estimates)
    treatment_balance_weighting(household_weights)


def download_and_process_acs_data() -> pl.DataFrame:
    """
    Load the 2023 ACS 1-year housing data.

    If the data is not already on disk, downloads the Census 2023 1-year ACS PUMS data
    and unzips it. It then reads it using Polars, and extracts household weights.

    Returns:
        polars.DataFrame: DataFrame containing household weights

    """
    temp_dir = Path("acs_data")
    if not Path(temp_dir / "psam_husa.csv").exists() or (
        not Path(temp_dir / "psam_husb.csv").exists()
    ):
        url = "https://www2.census.gov/programs-surveys/acs/data/pums/2023/1-Year/csv_hus.zip"
        print("Downloading ACS data...")
        response = requests.get(url, stream=True)

        if response.status_code != 200:
            raise Exception(
                f"Failed to download ACS data: Status code {response.status_code}"
            )

        # Create a BytesIO object from the response content
        zip_content = BytesIO(response.content)

        print("Download complete. Extracting files...")
        # Open the zip file
        with zipfile.ZipFile(zip_content) as zip_ref:
            # Find the CSV file (typically named like 'psam_husa.csv')
            csv_files = [f for f in zip_ref.namelist() if f.endswith(".csv")]

            if not csv_files:
                raise Exception("No CSV files found in the zip archive")

            dfs: list[pl.LazyFrame] = []
            for household_file in csv_files:
                print(f"Processing file: {household_file}")

                # Extract the file to a temporary location
                temp_dir.mkdir(parents=True)
                zip_ref.extract(household_file, temp_dir)

    # Read the CSV file with Polars
    print("Reading data with Polars...")
    dfs = [pl.scan_csv(x) for x in Path("acs_data").glob("*.csv")]

    df = pl.concat(dfs, how="vertical")
    # Select only the household weight column (WGTP)
    weights_df = (
        df.select("SERIALNO", "STATE", "WGTP", "BDSP", "NP", "HHT2", "HHL", "HINCP")
        .cast({"BDSP": pl.Float64, "HINCP": pl.Float64})
        .collect()
    )
    weights_df = weights_df.to_dummies(["HHT2", "HHL"])
    states_dict = {
        "Alabama": 1,
        "Alaska": 2,
        "Arizona": 4,
        "Arkansas": 5,
        "California": 6,
        "Colorado": 8,
        "Connecticut": 9,
        "Delaware": 10,
        "District of Columbia": 11,
        "Florida": 12,
        "Georgia": 13,
        "Hawaii": 15,
        "Idaho": 16,
        "Illinois": 17,
        "Indiana": 18,
        "Iowa": 19,
        "Kansas": 20,
        "Kentucky": 21,
        "Louisiana": 22,
        "Maine": 23,
        "Maryland": 24,
        "Massachusetts": 25,
        "Michigan": 26,
        "Minnesota": 27,
        "Mississippi": 28,
        "Missouri": 29,
        "Montana": 30,
        "Nebraska": 31,
        "Nevada": 32,
        "New Hampshire": 33,
        "New Jersey": 34,
        "New Mexico": 35,
        "New York": 36,
        "North Carolina": 37,
        "North Dakota": 38,
        "Ohio": 39,
        "Oklahoma": 40,
        "Oregon": 41,
        "Pennsylvania": 42,
        "Rhode Island": 44,
        "South Carolina": 45,
        "South Dakota": 46,
        "Tennessee": 47,
        "Texas": 48,
        "Utah": 49,
        "Vermont": 50,
        "Virginia": 51,
        "Washington": 53,
        "West Virginia": 54,
        "Wisconsin": 55,
        "Wyoming": 56,
    }
    weights_df = weights_df.join(
        pl.DataFrame({"state_name": states_dict.keys(), "STATE": states_dict.values()}),
        how="left",
        on="STATE",
    )
    weights_df = weights_df.select(
        "SERIALNO",
        "state_name",
        "WGTP",
        "STATE",
        "BDSP",
        "NP",
        "^HHT2.*$",
        "^HHL.*$",
        "HINCP",
    )
    print(f"Successfully extracted household weights. Shape: {weights_df.shape}")

    return weights_df


def download_state_housing_estimates() -> pl.DataFrame:
    """
    Download the 2023 state-level housing unit counts.

    Sourced from the U.S. Census Population Estimates program.

    Returns:
        polars.DataFrame: DataFrame with state housing unit counts

    """
    temp_dir = Path("popest_housing_data")
    housing_excel = Path(temp_dir / "popest_housing_counts.xlsx")
    if not housing_excel.exists():
        url = "https://www2.census.gov/programs-surveys/popest/tables/2020-2023/housing/totals/NST-EST2023-HU.xlsx"

        print("\nDownloading state-level housing unit estimates...")
        response = requests.get(url)

        if response.status_code != 200:
            raise Exception(
                f"Failed to download housing estimates data: Status code {response.status_code}"
            )

        print("Download complete. Processing state housing unit data...")
        temp_dir.mkdir(parents=True)
        with housing_excel.open("wb") as f:
            f.write(BytesIO(response.content).getbuffer())

    housing_estimates = pl.read_excel(housing_excel)

    housing_2023 = housing_estimates.drop_nulls("__UNNAMED__1").filter(
        pl.col("^table.*$").str.starts_with(".")
    )
    housing_2023 = housing_2023.rename(
        {housing_2023.columns[0]: "state", "__UNNAMED__5": "2023"}
    ).select("state", "2023")
    housing_2023 = housing_2023.with_columns(pl.col("state").str.slice(1, None))
    print(
        f"Successfully extracted state housing unit estimates. Shape: {housing_2023.shape}"
    )

    return housing_2023


def survey_weighting(df: pl.DataFrame, df_moments: pl.DataFrame) -> None:
    """
    Run the reweighting and analyze results.

    Pre-re-weighting, the housing counts by state should already be extremely close
    (by construction), don't take this as anything but an example of reweighting
    to some external data source.
    """
    df = (
        df.filter(pl.col("WGTP") > 0)
        .with_columns(pl.lit(1).alias("pop"))
        .sort("state_name")
    )
    state_group_xs = df.select("pop", "state_name").partition_by(
        "state_name", include_key=False
    )
    x = sp.block_diag(
        [x.to_numpy().astype(np.float64) for x in state_group_xs], format="csr"
    )
    moments = (
        df_moments.sort("state").select("2023").to_numpy().astype(np.float64).ravel()
    )
    weights0 = df.select("WGTP").to_numpy().astype(np.float64).ravel()

    print("ACS 2023 1-year weighted housing counts:")
    print((sp.diags_array(weights0) @ x).sum(0))
    print("Pop Estimates 2023 state housing counts:")
    print(moments)
    print("Total difference, Pop estimates - Old ACS Counts:")
    print(np.sum(np.abs((sp.diags_array(weights0) @ x).sum(0) - moments)))
    n = x.shape[0]
    w0_scale = np.mean(weights0)
    moments = moments / w0_scale
    weights0 = weights0 / w0_scale
    mean_population_moments = moments / n
    res = ebw.entropy_balance(
        mean_population_moments=mean_population_moments, x_sample=x, weights0=weights0
    )
    print("Reweighted ACS counts:")
    new_acs_counts = (sp.diags_array(res.new_weights * w0_scale) @ x).sum(0)
    print(new_acs_counts)
    print("Total difference, Pop estimates - New ACS Counts:")
    print(np.sum(np.abs(moments * w0_scale - new_acs_counts)))
    print()
    print("Correlation (original weights, new weights):")
    print(np.corrcoef(res.new_weights, weights0)[0, 1])
    print("Max/min new weight ratios:")
    print(np.max(res.new_weights / weights0), np.min(res.new_weights / weights0))


def treatment_balance_weighting(df: pl.DataFrame) -> None:
    """
    Reweight a set of samples to all have the same weighted characteristics.

    I calculate the original weighted national sample average of # of bedrooms,
    # of people, dummies of category of household type, dummies of categories
    of language spoken at home, and houshold income.
    I then reweight each state so that the within-state weighted averages of
    all categories are equal to the national average.

    As this is just an example, I don't deal at all with missing data to make the
    data construction simpler.
    """
    df = df.filter(pl.col("WGTP").gt(0.0)).drop_nulls()
    df = df.with_columns(
        pl.lit(1.0).alias("count"),
        pl.col("WGTP"),
        pl.col("WGTP").sum().over("STATE").alias("state_wgtp"),
    )
    df = df.sort("STATE", "SERIALNO")
    df = df.drop("^.*_null$")
    X_cols = pl.col("count", "BDSP", "NP", "^HHT2.*$", "^HHL.*$", "HINCP")
    overall_averages = df.select(
        X_cols * (pl.col("WGTP") / pl.col("WGTP").mean())
    ).mean()

    state_specific_x = [
        x.select(X_cols).to_numpy().astype(np.float64) for x in df.partition_by("STATE")
    ]

    # Initial weights should sum to 1 within state to be consistent with the RHS average
    weights0 = df.select(pl.col("WGTP") / pl.col("state_wgtp")).to_numpy().ravel()
    x_sample = sp.block_diag(state_specific_x, format="csr")
    moments = np.tile(overall_averages.to_numpy().ravel(), 51)

    res = ebw.entropy_balance(
        x_sample=x_sample, mean_population_moments=moments, weights0=weights0
    )
    print(f"Converged? {res.converged}")

    # Rescaled new weights and within-state averages are all equal to overall averages
    new_cv = (sp.diags_array(res.new_weights / weights0.sum()) @ x_sample).sum(
        0
    ) - moments
    print(f"Overall moments miss post-reweight: {new_cv}")
    alaska_weights = weights0[df["STATE"].to_numpy() == 2]
    new_alaska_weights = res.new_weights[df["STATE"].to_numpy() == 2]
    alaska_orig_moments = (
        sp.diags_array(alaska_weights) @ state_specific_x[1]
    ).sum(0)
    alaska_new_moments = (
        sp.diags_array(new_alaska_weights / weights0.sum()) @ state_specific_x[1]
    ).sum(0)
    df_alaska =  pl.from_numpy(
        np.c_[alaska_orig_moments, alaska_new_moments],
        schema=["alaska_orig", "alaska_new"],
    )
    pl.Config.set_tbl_rows(50)
    compare_df = (
        pl.concat([overall_averages.unpivot().rename({"value": "national_avg"}), df_alaska], how="horizontal")
    )
    print(f"E.g. Alaska moment comparison: {compare_df}")
    print("Correlation (original weights, new weights):")
    print(np.corrcoef(res.new_weights, weights0)[0, 1])




if __name__ == "__main__":
    main()
