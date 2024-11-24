import pandas as pd
import streamlit as st
import markdown2
import os
import asyncio
from playwright.async_api import async_playwright
import pathlib
import os
import time
import seaborn as sns
import matplotlib.pyplot as plt
from Visualization.HelperFun import Axis_Limits, Column_filter, Column_Remover
from Visualization.SambavAI import Generator_Code_Sambav_AI
from Visualization.HelperFun import Column_filter, Axis_Limits, find_repeating_categorical_columns, Column_Remover, copy_file, delete_files

files_to_delete = ["SampleScatterPlot_copy.html", "SampleScatterPlot_copy.md","SampleScatterPlot_copy.pdf","BoxPlt_copy.md","BoxPlt_copy.pdf","BoxPlt_copy.html"]

    
#? =================================== Additonal Functions ===================================

import markdown2
import os
import asyncio
from playwright.async_api import async_playwright
import pathlib
import os

import shutil

def copy_file(src_path):
    # Generate the destination path by appending '_copy' before the file extension
    dest_path = src_path.replace(".", "_copy.", 1)
    
    # Copy the file
    shutil.copy(src_path, dest_path)
    
    return dest_path




def replace_all_markers(file_path, marker_content_dict, end_marker=None):
    """
    Replace multiple sections of content in a Markdown file, based on a dictionary of start markers and new content.
    
    Args:
    - file_path: The path to the Markdown file.
    - marker_content_dict: A dictionary where the keys are start markers and the values are lists of new content.
    - end_marker: Optional. If provided, content between each start_marker and end_marker will be replaced.
    
    Example usage:
    replace_all_markers(
        "output.md", 
        {
            "## Existing Section 1": ["## New Section 1\n", "New content 1 here\n"],
            "## Existing Section 2": ["## New Section 2\n", "New content 2 here\n"]
        }, 
        "## Another Section"
    )
    """
    # Read the existing content of the file
    file_path = copy_file(file_path)
    with open(file_path, "r") as md_file:
        lines = md_file.readlines()  # Read all lines into a list

    for start_marker, new_content in marker_content_dict.items():
        i = 0
        while i < len(lines):
            # Find the index of the start marker
            if start_marker in lines[i]:
                start_index = i

                # If an end marker is provided, find the index of the end marker
                if end_marker:
                    end_index = None
                    for j in range(start_index + 1, len(lines)):
                        if end_marker in lines[j]:
                            end_index = j
                            break

                    # If end_marker is not found, replace content up to the end of the file
                    if end_index is None:
                        end_index = len(lines)
                else:
                    # If no end_marker is provided, replace only the start_marker line
                    end_index = start_index + 1

                # Replace the content between the start and end markers
                lines[start_index:end_index] = new_content

                # Move index past the newly inserted content
                i = start_index + len(new_content)
            else:
                i += 1

    # Write the updated content back to the file
    with open(file_path, "w") as md_file:
        md_file.writelines(lines)
    

def convert_md_to_html(input_md_file: str) -> str:
    """
    Convert a Markdown file to an HTML file.

    Args:
        input_md_file (str): Path to the input Markdown file.

    Returns:
        str: Path to the output HTML file created with the same name as the input file.
    """
    try:
        # Read the Markdown file
        with open(input_md_file, "r", encoding='utf-8') as md_file:
            markdown_content = md_file.read()

        # Convert Markdown to HTML
        html_content = markdown2.markdown(markdown_content)

        # Derive the output HTML file path
        output_html_file = os.path.splitext(input_md_file)[0] + ".html"

        # Write the HTML content to a new file
        with open(output_html_file, "w", encoding='utf-8') as html_file:
            html_file.write(html_content)

        print(f"HTML file '{output_html_file}' created successfully!")
        return output_html_file  # Return the path of the generated HTML file

    except FileNotFoundError:
        print(f"Error: The file '{input_md_file}' does not exist.")
    except Exception as e:
        print(f"An error occurred: {e}")

    return None  # Return None in case of an error



async def generate_pdf_from_html(html_file: str):
    # Convert the provided HTML file path into an absolute path
    filePath = os.path.abspath(html_file)
    # Derive the URL path of the local HTML file to be opened in the browser
    fileUrl = pathlib.Path(filePath).as_uri()
    
    # Create the output PDF file name based on the HTML file name
    pdf_file_path = os.path.splitext(filePath)[0] + ".pdf"  # Change extension to .pdf

    async with async_playwright() as p:
        # Create a browser instance
        browser = await p.chromium.launch()
        # Open a new tab in the browser
        page = await browser.new_page()
        # Go to the URL of the HTML page
        await page.goto(fileUrl)
        # Change CSS media type to screen
        await page.emulate_media(media="screen")
        # Print the HTML page as a PDF in the browser
        await page.pdf(path=pdf_file_path, format="A4", landscape=False, margin={"top": "2cm"})
        # Close the browser
        await browser.close()
    
    return pdf_file_path  # Return the path of the generated PDF

def run_html_to_pdf_conversion(html_file):
    try:
        # Get the current event loop
        loop = asyncio.get_event_loop()
        generated_pdf = loop.run_until_complete(generate_pdf_from_html(html_file))
        print(f"PDF generated: {generated_pdf}")
        return generated_pdf
    except RuntimeError as e:
        if str(e) == "asyncio.run() cannot be called from a running event loop":
            # This case might not be necessary anymore, as we are using the event loop directly
            pass
        else:
            raise
        
import os
import pathlib
import asyncio
from playwright.async_api import async_playwright
from multiprocessing import Process, Queue

def generate_pdf_process(queue, html_file: str):
    """Target function for multiprocessing to generate PDF."""
    pdf_file_path = os.path.splitext(html_file)[0] + ".pdf"
    
    # Run Playwright in this function
    async def run_playwright():
        filePath = os.path.abspath(html_file)
        fileUrl = pathlib.Path(filePath).as_uri()

        async with async_playwright() as p:
            browser = await p.chromium.launch()
            page = await browser.new_page()
            await page.goto(fileUrl)
            await page.emulate_media(media="screen")
            await page.pdf(path=pdf_file_path, format="A4", landscape=False, margin={"top": "2cm"})
            await browser.close()

        return pdf_file_path

    # Run the coroutine in a new event loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    pdf_path = loop.run_until_complete(run_playwright())
    queue.put(pdf_path)  # Send the result back to the main process

def generate_pdf_with_multiprocessing(html_file: str):
    queue = Queue()
    process = Process(target=generate_pdf_process, args=(queue, html_file))
    process.start()
    process.join()  # Wait for the process to finish
    pdf_file_path = queue.get()  # Get the result from the queue
    return pdf_file_path


#? ================================================= Box Plot ================================================= 
 

def find_repeating_categorical_columns(df):
    """
    This function finds the categorical columns in a DataFrame that contain repeating values
    and excludes string columns with only unique values.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame.
    
    Returns:
    list: A list of column names that are categorical and have repeating values.
    """
    # Step 1: Identify categorical columns (object or category types)
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns
    
    # Step 2: Filter for columns with repeating values
    repeating_categorical_cols = []
    
    for col in categorical_columns:
        # Check if there are any duplicated values in the column
        if df[col].duplicated().any():
            repeating_categorical_cols.append(col)
    
    return repeating_categorical_cols

def Y_column(df, x_column):
    """
    This function selects a Y-axis column based on the data type of the X-axis column.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame.
    x_column (pd.Series): The selected X-axis column.
    
    Returns:
    y_column (str): The selected Y-axis column if applicable, or None if no selection is made.
    """
    # Check if x_column is a categorical type (object or category)
    x_column = x_column[0]
    if pd.api.types.is_object_dtype(x_column) or pd.api.types.is_categorical_dtype(x_column):
        # If X is categorical, Y should be numeric
        y_column = st.sidebar.selectbox("📊 Select Y-axis column", Column_filter(df, 'number'))
        return y_column
    # Check if x_column is numeric (int or float)
    elif pd.api.types.is_numeric_dtype(x_column):
        # No selection for Y-axis if X is numeric (you can adjust this behavior)
        return None
    else:
        return None
    
def X_Column(df):
    categorical_col = find_repeating_categorical_columns(df)
    numerical_col = Column_filter(df, 'number')
    return categorical_col + numerical_col
     

def create_box_plot(data, x_col, y_col, hue_col, palette, showfliers, outlier_marker, outlier_size, outlier_color, 
                    linewidth, line_color, bg_color, showmeans, meanColor, meanMarker, figsize, 
                    xlim, ylim, x_label, y_label, title):
    plt.figure(figsize=figsize)

    # Set Seaborn style and background color
    sns.set(style="whitegrid")
    plt.gca().set_facecolor(bg_color)

    # Create the boxplot
    sns.boxplot(
        data=data,
        x=x_col,
        y=y_col,
        hue=hue_col if hue_col else None,  # Use hue_col only if not None
        palette=palette,
        showfliers=showfliers,
        showmeans=showmeans,
        flierprops={'marker': outlier_marker, 'color': outlier_color, 'markersize': outlier_size} if showfliers else None,
        meanprops={'marker': meanMarker, 'markersize': 10, 'markerfacecolor': meanColor, 'markeredgecolor': meanColor} if showmeans else None,
        linewidth=linewidth,
        whiskerprops=dict(color=line_color, linewidth=linewidth),
        capprops=dict(color=line_color, linewidth=linewidth),
        medianprops=dict(color=line_color, linewidth=linewidth),
    )

    # Set axis limits if defined
    if xlim:
        plt.xlim(xlim)
    if ylim:
        plt.ylim(ylim)

    # Apply custom titles and labels
    plt.title(title, fontsize=18, fontweight='bold', color="#444444")
    if x_label:
        plt.xlabel(x_label)
    if y_label:
        plt.ylabel(y_label)

    # Legend customization if hue_col is used
    if hue_col:
        plt.legend(loc='upper center', ncol=5, bbox_to_anchor=(0.5, -0.15), fancybox=True, framealpha=0.9, facecolor='lightgray', edgecolor='black')

    # Add grid
    plt.grid(True, which='both', linestyle='--', linewidth=0.6, color='gray', alpha=0.7)

    # Display plot
    st.pyplot(plt)

    # Save plot and clear figure
    plt.tight_layout()
    plt.savefig("box_plot.png")
    plt.clf()  

def create_box_plots_Multi_Column(data, selected_columns, hue_col=None, palette='Set3', showfliers=True, outlier_marker='o', outlier_size=5,
                     outlier_color='#Ff0000', linewidth=2, line_color='#0000ff', bg_color='#F5F5F5', showmeans=False,
                     meanColor='#Ffa500', meanMarker='D', figsize=(12, 6), x_label=None, 
                     y_label='Values', title='Box Plot'):
    
    # Number of selected columns
    num_columns = len(selected_columns)
    
    # Calculate the number of rows needed, 2 columns per row
    num_rows = (num_columns + 1) // 2

    # Create subplots dynamically based on selected columns
    fig, axes = plt.subplots(num_rows, 2, figsize=figsize, constrained_layout=True)

    # Flatten axes array for easier iteration
    axes = axes.flatten()

    # Set Seaborn style
    sns.set(style="whitegrid")

    # Plot each column's boxplot
    for i, col in enumerate(selected_columns):
        # Set background color
        axes[i].set_facecolor(bg_color)

        # Create the boxplot for each column
        sns.boxplot(
            data=data,
            x=col,
            ax=axes[i],
            palette=palette if hue_col else None,  # Use palette only if hue_col is set
            showfliers=showfliers,
            showmeans=showmeans,
            flierprops={'marker': outlier_marker, 'color': outlier_color, 'markersize': outlier_size} if showfliers else None,
            meanprops={'marker': meanMarker, 'markersize': 10, 'markerfacecolor': meanColor, 'markeredgecolor': meanColor} if showmeans else None,
            linewidth=linewidth,
            whiskerprops=dict(color=line_color, linewidth=linewidth),
            capprops=dict(color=line_color, linewidth=linewidth),
            medianprops=dict(color=line_color, linewidth=linewidth),
        )

        # Set titles and labels for each subplot
        axes[i].set_title(f'Boxplot of {col}', fontsize=14, fontweight='bold')
        axes[i].set_xlabel(x_label[i] if x_label else col)
        axes[i].set_ylabel(y_label)

        

    # Hide any extra axes if the number of columns is odd
    if num_columns % 2 != 0:
        axes[-1].set_visible(False)

    # Set the overall plot title
    plt.suptitle(title, fontsize=18, fontweight='bold', color="#444444", y=1.05)

    # Show the plot using Streamlit
    st.pyplot(fig)

    # Save plot and clear figure
    plt.savefig("box_plots.png")
    plt.clf()


def Box_plot_visualize(df):
    st.sidebar.markdown("""<h2 style='color: #FFFF4D; font-weight: bold;font-size:18px;'>A. 📊 Axis and Grid Customization</h2>""", unsafe_allow_html=True)
    x_column = st.sidebar.multiselect("📊 Select X-axis column", X_Column(df), default=X_Column(df)[:1])
    x_col_len = len(x_column)
    if x_col_len == 1:
        x_column = x_column[0]
        y_column = Y_column(df,x_column)
    else:
        y_column = None
    hue_column = st.sidebar.selectbox("📊 Select Hue column (optional)", [None] + Column_filter(df, 'object'))

    # Outlier customization
    st.sidebar.markdown("""<h2 style='color: #FFFF4D; font-weight: bold;font-size:18px;'>B. 📊 Outlier Customization</h2>""", unsafe_allow_html=True)
    outlier_mark = ['+','D', 'o', '*']
    showfliers = st.sidebar.checkbox("Show Outliers", value=True)
    outlier_marker = st.sidebar.selectbox("Select Outlier Marker", outlier_mark)
    outlier_size = st.sidebar.slider("Select Outlier Marker Size", min_value=1, max_value=10, value=5)
    outlier_color = st.sidebar.color_picker("Pick Outlier Marker Color", "#Ff0000")

    # Mean customization
    st.sidebar.markdown("""<h2 style='color: #FFFF4D; font-weight: bold;font-size:18px;'>C. 📊 Mean Customization</h2>""", unsafe_allow_html=True)
    showmeans = st.sidebar.checkbox("Show Mean", value=True)
    meanMarker = st.sidebar.selectbox("Select Mean Marker", outlier_mark)
    meanColor = st.sidebar.color_picker("Pick Mean Marker Color", "#Ffa500")

    # Line customization
    st.sidebar.markdown("""<h2 style='color: #FFFF4D; font-weight: bold;font-size:18px;'>D. 🎨 Line customization</h2>""", unsafe_allow_html=True)
    linewidth = st.sidebar.slider("Select Line Width", min_value=1.0, max_value=3.0, value=2.0)
    line_color = st.sidebar.color_picker("Pick Line Color", "#0000ff")

    # Box customization
    st.sidebar.markdown("""<h2 style='color: #FFFF4D; font-weight: bold;font-size:18px;'>E. 🎨 Box customization</h2>""", unsafe_allow_html=True)
    plot_width = st.sidebar.slider("Select plot width (in inches)", min_value=5, max_value=20, value=14, step=1)
    plot_height = st.sidebar.slider("Select plot height (in inches)", min_value=5, max_value=20, value=8, step=1)
    palette = st.sidebar.selectbox("Select Color Palette", ["Set3","deep", "pastel", "dark", "colorblind", "viridis", "rocket", "mako", "flare"])
    bg_color = st.sidebar.color_picker("Pick Background Color", "#F5F5F5")

    # Axis limits
    if x_column and x_col_len == 1:
        xlim = Axis_Limits(df, x_column, 'x')
    else:
        xlim = None
        
    if y_column and x_col_len == 1:
        ylim = Axis_Limits(df, y_column, 'y')
    else:
        ylim = None

    if x_col_len == 1:
        # Axis title customization
        st.sidebar.markdown("""<h2 style='color: #FFFF4D; font-weight: bold;font-size:18px;'>F. 📊 Axis Title Customization</h2>""", unsafe_allow_html=True)
        x_label = st.sidebar.text_input("X-axis Label (optional)", x_column if x_column else "") 
        y_label = st.sidebar.text_input("Y-axis Label (optional)", y_column if y_column else "") if y_column else None
        title = st.sidebar.text_input("Plot Title (optional)", "Box Plot")
    else:
        # get the label for the x-axis
        st.sidebar.markdown("""<h2 style='color: #FFFF4D; font-weight: bold;font-size:18px;'>F. 📊 Axis Title Customization</h2>""", unsafe_allow_html=True)
        y_label = st.sidebar.text_input("Y-axis Label (optional)", y_column if y_column else "") if y_column else None
        x_label = x_column
        title = st.sidebar.text_input("Plot Title (optional)", "Box Plot")
        

    # Ensure hue_column is None if the user selects "None"
    hue_column = hue_column if hue_column != "None" else None
    
    
    if x_col_len == 1:
        if st.sidebar.button("Generate Box Plot"):
            with st.spinner("Generating Box Plot..."):
                create_box_plot(data=df, x_col=x_column, y_col=y_column, hue_col=hue_column, palette=palette, showfliers=showfliers, outlier_marker=outlier_marker, outlier_size=outlier_size, outlier_color=outlier_color, 
                            linewidth=linewidth, line_color=line_color, bg_color=bg_color, showmeans=showmeans, meanColor=meanColor, meanMarker=meanMarker, figsize=(plot_width,plot_height),
                            xlim=xlim, ylim=ylim, x_label=x_label, y_label=y_label, title=title)
                
                with open('box_plot.png', "rb") as file:
                    st.download_button(
                        label="Download Box Plot",
                        data=file,
                        file_name="box_plot.png",
                        mime="image/png"
                    )

                st.success("Box plot generated successfully!")
        if st.sidebar.button("Generate Code"):
            with st.spinner("Generating Code..."):
                create_box_plot(data=df, x_col=x_column, y_col=y_column, hue_col=hue_column, palette=palette, showfliers=showfliers, outlier_marker=outlier_marker, outlier_size=outlier_size, outlier_color=outlier_color, 
                        linewidth=linewidth, line_color=line_color, bg_color=bg_color, showmeans=showmeans, meanColor=meanColor, meanMarker=meanMarker, figsize=(plot_width,plot_height),
                        xlim=xlim, ylim=ylim, x_label=x_label, y_label=y_label, title=title)
                with st.spinner("Generating Code..."):
                    Code = Generator_Code(df, x_column, y_column, hue_column, palette, showfliers, outlier_marker, outlier_size, outlier_color, linewidth, line_color, bg_color, showmeans, meanColor, meanMarker, (plot_width,plot_height), xlim, ylim, x_label, y_label, title)
                
                st.code(Code, language='python')
                    
                col1, col2 = st.columns([1,1])
                
                with col1:
                    with open('box_plot.png', "rb") as file:
                        st.download_button(
                            label="Download Box Plot",
                            data=file,
                            file_name="box_plot.png",
                            mime="image/png"
                        )
                with col2:
                    changing_sec = {
                        "ImagesofBoxPlot": "box_plot.png",
                        "CodingPart": Code
                    }
                    replace_all_markers("SampleBoxPlot.md", changing_sec)
                    # Convert the Markdown file to HTML
                    html_file = convert_md_to_html("SampleBoxPlot_copy.md")
                    # Generate PDF
                    with st.spinner("Generating PDF..."):
                        pdf_file = generate_pdf_with_multiprocessing(html_file)

                    st.success("PDF generated successfully!")

                    # Add a download button for the generated PDF
                    with open(pdf_file, "rb") as file:
                        st.download_button(
                            label="Download PDF",
                            data=file,
                            file_name="SampleBoxPlot.pdf",
                            mime="application/pdf"
                        )     
        else:
            pass
    else:
        if st.sidebar.button("Generate Box Plot"):
            with st.spinner("Generating Box Plot..."):
                
                create_box_plots_Multi_Column(data=df, selected_columns=x_column, hue_col=hue_column, palette=palette, showfliers=showfliers, outlier_marker=outlier_marker, outlier_size=outlier_size,
                        outlier_color=outlier_color, linewidth=linewidth, line_color=line_color, bg_color=bg_color, showmeans=showmeans,
                        meanColor=meanColor, meanMarker=meanMarker, figsize=(plot_width,plot_height), x_label=x_label, 
                        y_label=y_label, title=title)
                
                with open('box_plots.png', "rb") as file:
                    st.download_button(
                        label="Download Box Plot",
                        data=file,
                        file_name="box_plot.png",
                        mime="image/png"
                    )

                st.success("Box plot generated successfully!")
        
        if st.sidebar.button("Generate Code & Get PDF"):
            with st.spinner("Generating Code..."):
                create_box_plots_Multi_Column(data=df, selected_columns=x_column, hue_col=hue_column, palette=palette, showfliers=showfliers, outlier_marker=outlier_marker, outlier_size=outlier_size,
                        outlier_color=outlier_color, linewidth=linewidth, line_color=line_color, bg_color=bg_color, showmeans=showmeans,
                        meanColor=meanColor, meanMarker=meanMarker, figsize=(plot_width,plot_height), x_label=x_label, 
                        y_label=y_label, title=title)
                with st.spinner("Generating Code..."):
                    Code = Generator_Code_Multi_Plot(df, x_column,hue_column, palette, showfliers, outlier_marker, outlier_size, outlier_color, linewidth, line_color, bg_color, showmeans, meanColor, meanMarker, (plot_width,plot_height), x_label, y_label, title)
                st.code(Code, language='python')
                
                col1, col2 = st.columns([1,1])
                
                with col1:
                    with open('box_plots.png', "rb") as file:
                        st.download_button(
                            label="Download Box Plot",
                            data=file,
                            file_name="box_plots.png",
                            mime="image/png"
                        )
                with col2:
                    changing_sec = {
                        "ImagesofBoxPlot": "box_plots.png",
                        "CodingPart": Code
                    }
                    replace_all_markers("SampleBoxPlot.md", changing_sec)
                    # Convert the Markdown file to HTML
                    html_file = convert_md_to_html("SampleBoxPlot_copy.md")

                    # Generate PDF
                    with st.spinner("Generating PDF..."):
                        pdf_file = generate_pdf_with_multiprocessing(html_file)

                    st.success("PDF generated successfully!")

                    # Add a download button for the generated PDF
                    with open(pdf_file, "rb") as file:
                        st.download_button(
                            label="Download PDF",
                            data=file,
                            file_name="SampleBoxPlot.pdf",
                            mime="application/pdf",
                            on_click=lambda: delete_files(files_to_delete)
                        )  
        else:
            pass

    
        
        
        

def Generator_Code(df, x_column, y_column, hue_column, palette, showfliers, outlier_marker, outlier_size, outlier_color, linewidth, line_color, bg_color, showmeans, meanColor, meanMarker, figsize, xlim, ylim, x_label, y_label, title):

    prompt = f"""
    Create a box plot using the following parameters in python language:
    1. Use the dataset stored in the variable 'df'.
    2. For the x-axis, use the column '{x_column}'.
    3. For the y-axis, use the column '{y_column}'.
    4. {f"Group the data by the '{hue_column}' column and apply different colors." if hue_column else "Do not group the data by any hue column."}
    5. Use the color palette '{palette}'.
    6. {f"Display outliers with marker '{outlier_marker}', size '{outlier_size}', and color '{outlier_color}'." if showfliers else "Do not show outliers."}
    7. Set the thickness of the box plot lines to {linewidth} and the line color to '{line_color}'.
    8. Set the background color of the plot to '{bg_color}'.
    9. {f"Display the mean with marker '{meanMarker}' and color '{meanColor}'." if showmeans else "Do not display the mean."}
    10. Set the figure size to {figsize}.
    11. {f"Limit the x-axis to {xlim}." if xlim else "Do not set x-axis limits."}
    12. {f"Limit the y-axis to {ylim}." if ylim else "Do not set y-axis limits."}
    13. Label the x-axis as '{x_label}'.
    14. Label the y-axis as '{y_label}'.
    15. Set the title of the plot to '{title}'.
    """
    Code = Generator_Code_Sambav_AI(prompt)
    return Code

def Generator_Code_Multi_Plot(df, selected_columns, hue_column, palette, showfliers, outlier_marker, outlier_size, outlier_color,
                   linewidth, line_color, bg_color, showmeans, meanColor, meanMarker, figsize, 
                   x_label, y_label, title):
    # Generating the prompt for multiple box plots
    prompt = f"""
    Create multiple box plots using the following parameters in python language. I am providing you a varibles to customize the box plot:
    
    1. Datastore in the variable 'df'.
    2. X_axis of subplots are {selected_columns}.
    3. {f"Group the data by the '{hue_column}' column and apply different colors." if hue_column else "Do not group the data by any hue column."}
    4. Use the color palette '{palette}' for the box plots.
    5. {f"Display outliers with marker '{outlier_marker}', size '{outlier_size}', and color '{outlier_color}'." if showfliers else "Do not show outliers."}
    6. Set the thickness of the box plot lines to {linewidth} and the line color to '{line_color}'.
    7. Set the background color of each subplot to '{bg_color}'.
    8. {f"Display the mean with marker '{meanMarker}' and color '{meanColor}'." if showmeans else "Do not display the mean."}
    9. Set the figure size to {figsize}.
    10. Label the x-axis of each subplot with the name of the respective column being plotted.
    11. Use the common y-axis label '{y_label}' for all subplots.
    12. Set the title for each subplot based on their name to 'Boxplot of ...', where ... is the name of the respective column.
    13. Set the overall title of the entire plot to '{title}'.
    """
    Code = Generator_Code_Sambav_AI(prompt)
    
    return Code

