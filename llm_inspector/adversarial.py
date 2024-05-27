import pandas as pd
import datetime
from configparser import ConfigParser


class Adversarial:

    def __init__(
        self,
        config,
        outfilepath=None,
        capability=None,
        subcapability=None,
        samplesize=None,
    ):
        adversarial_file = config["Adversarial_File"]

        if capability is None:
            capability = config.get("Adversarial_File", "capability")

        if subcapability is None:
            subcapability = config.get("Adversarial_File", "subcapability")

        dt_time = datetime.datetime.now()
        self.inputfile = (
            adversarial_file["Adversarial_input_FilePath"]
            + adversarial_file["Adversarial_FileName"]
        )
        adversary_df = pd.read_excel(self.inputfile)
        self.adversary_df1 = adversary_df

        self.output_path = (
            adversarial_file["Adversarial_output_path"]
            + adversarial_file["Adversarial_Output_fileName"]
            + str(capability)
            + "_"
            + str(dt_time.year)
            + str(dt_time.month)
            + str(dt_time.day)
            + "__"
            + str(dt_time.hour)
            + str(dt_time.minute)
            + ".xlsx"
        )
        self.adv_output = outfilepath if outfilepath is not None else self.output_path

        if capability.lower() == "all":
            capability = None

        if subcapability.lower() == "all":
            subcapability = None

        self.samplesize = samplesize if samplesize is not None else 1000

        if capability is not None:
            self.config_capability = capability 
        else:
            self.config_capability = []
        
        if subcapability is not None:
            self.config_subcapability = subcapability  
        else:
            self.config_subcapability = []

    def random_adversarial_selection(self):
        # Function to return Random adversarial TestData using sample Size or default 1000 records

        numrows = self.samplesize
        self.sample_record_df = self.adversary_df1.sample(n=numrows)

        df_melt = self.sample_record_df.melt(
            id_vars=["Capability", "Sub Capability", "Prompt"]
        )
        self.transformed_records_df = df_melt.filter(
            ["Capability", "Sub Capability", "Prompt", "value"]
        )

        self.transformed_records_df.rename(columns={"value": "Char Len"}, inplace=True)

        return self.transformed_records_df

    def export_adversarial_data(self):

        if self.config_capability != [] and self.config_subcapability == []:
            self.filtered_cap_subcap_df = self.adversary_df1[
                self.adversary_df1["Capability"].str.lower() == self.config_capability.lower()
            ]

        elif self.config_capability == [] and self.config_subcapability != []:
            self.filtered_cap_subcap_df = self.adversary_df1[
                self.adversary_df1["Sub Capability"].str.lower()
                == self.config_subcapability.lower()
            ]

        elif self.config_capability != [] and self.config_subcapability != []:
            self.filtered_cap_subcap_df = self.adversary_df1[
                (
                    self.adversary_df1["Capability"].str.lower()
                    == self.config_capability.lower()
                )
                & (
                    self.adversary_df1["Sub Capability"].str.lower()
                    == self.config_subcapability.lower()
                )
            ]
        elif self.config_capability == [] and self.config_subcapability == []:
            self.filtered_cap_subcap_df = self.random_adversarial_selection()

        df_melt = self.filtered_cap_subcap_df.melt(
            id_vars=["Capability", "Sub Capability", "Prompt"]
        )
        self.transformed_cap_subcap_df = df_melt.filter(
            ["Capability", "Sub Capability", "Prompt", "value"]
        )

        self.transformed_cap_subcap_df.rename(
            columns={"value": "Char Len"}, inplace=True
        )

        print(
            "capability :",
            (
                self.config_capability
                if self.config_capability is not None
                else self.transformed_cap_subcap_df["Capability"].unique()
            ),
        )

        print(
            "subcapability :",
            (
                self.config_subcapability
                if self.config_subcapability is not None
                else self.transformed_cap_subcap_df["Sub Capability"].unique()
            ),
        )

        print(
            "Generated adversarial test data can be obtained on the provided path:",
            self.adv_output,
        )

        self.transformed_cap_subcap_df.sort_values(
            ["Capability", "Sub Capability"]
        ).to_excel(self.adv_output)

        return self.transformed_cap_subcap_df
