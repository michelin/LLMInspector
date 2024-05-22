import pandas as pd
import datetime
from configparser import ConfigParser


class Adversarial:

    def __init__(
        self,
        config,
        outFilePath=None,
        capability=None,
        subCapability=None,
        sampleSize=None,
    ):
        # config = ConfigParser()
        # config.read(config_path)
        Adversarial_File = config["Adversarial_File"]

        capability = config.get("Adversarial_File", "capability")

        dt_time = datetime.datetime.now()
        self.inputFile = (
            Adversarial_File["Adversarial_input_FilePath"]
            + Adversarial_File["Adversarial_FileName"]
        )
        adversary_df = pd.read_excel(self.inputFile)
        self.adversary_df1 = adversary_df
        # self.outPath = os.path.join(sys.path[0], 'Output')

        self.output_path = (
            Adversarial_File["Adversarial_output_path"]
            + Adversarial_File["Adversarial_Output_fileName"]
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
        self.Adv_output = outFilePath if outFilePath is not None else self.output_path

        if capability.lower() == "all":
            capability = None

        subCapability = config.get("Adversarial_File", "subCapability")
        if subCapability.lower() == "all":
            subCapability = None

        self.sampleSize = sampleSize if sampleSize is not None else 1000

        capability_lst = self.adversary_df1["Capability"].unique()

        self.config_capability = capability
        self.config_subCapability = subCapability

        if capability is not None:
            self.capability = capability
        else:
            self.capability = []

        if subCapability is not None:
            self.subCapability = subCapability
        else:
            self.subCapability = []

    def randomAdversarial_Selection(self):
        # Function to return Random adversarial TestData using sample Size or default 1000 records

        numRows = self.sampleSize
        self.sample_record_df = self.adversary_df1.sample(n=numRows)

        df_melt = self.sample_record_df.melt(
            id_vars=["Capability", "Sub Capability", "Prompt"]
        )
        self.transformed_records_df = df_melt.filter(
            ["Capability", "Sub Capability", "Prompt", "value"]
        )

        self.transformed_records_df.rename(columns={"value": "Char Len"}, inplace=True)

        return self.transformed_records_df

    def export_adversarial_data(self):

        if self.capability != [] and self.subCapability == []:
            self.filtered_cap_subcap_df = self.adversary_df1[
                self.adversary_df1["Capability"].str.lower() == self.capability.lower()
            ]

        elif self.capability == [] and self.subCapability != []:
            self.filtered_cap_subcap_df = self.adversary_df1[
                self.adversary_df1["Sub Capability"].str.lower()
                == self.subCapability.lower()
            ]

        elif self.capability != [] and self.subCapability != []:
            self.filtered_cap_subcap_df = self.df1[
                (
                    self.adversary_df1["Capability"].str.lower()
                    == self.capability.lower()
                )
                & (
                    self.adversary_df1["Sub Capability"].str.lower()
                    == self.subCapability.lower()
                )
            ]
        elif self.capability == [] and self.subCapability == []:
            self.filtered_cap_subcap_df = self.randomAdversarial_Selection()

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
                self.capability
                if self.config_capability is not None
                else self.transformed_cap_subcap_df["Capability"].unique()
            ),
        )

        print(
            "subCapability :",
            (
                self.subCapability
                if self.config_subCapability is not None
                else self.transformed_cap_subcap_df["Sub Capability"].unique()
            ),
        )

        print(
            "Generated adversarial test data can be obtained on the provided path:",
            self.Adv_output,
        )

        self.transformed_cap_subcap_df.sort_values(
            ["Capability", "Sub Capability"]
        ).to_excel(self.Adv_output)

        return self.transformed_cap_subcap_df
