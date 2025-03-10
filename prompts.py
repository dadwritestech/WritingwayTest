import json
import os
from PyQt5.QtWidgets import (
    QDialog, QWidget, QVBoxLayout, QSplitter, QTreeWidget, QTreeWidgetItem,
    QTextEdit, QPushButton, QHBoxLayout, QMenu, QInputDialog, QMessageBox,
    QLabel, QComboBox, QSpinBox, QDoubleSpinBox, QApplication
)
from PyQt5.QtCore import Qt
from llm_api_aggregator import WWApiAggregator
from settings_manager import WWSettingsManager

def get_workshop_prompts(project_name):
    """
    Loads workshop prompts from a project-specific file.
    The file is expected to be named 'prompts_<projectname_no_spaces>.json'.
    Returns a list of prompt objects from the "Workshop" category.
    If the file is missing or fails to load, returns a dummy prompt.
    """
    base_name = f"prompts_{project_name.replace(' ', '')}"
    filename = f"{base_name}.json"
    if os.path.exists(filename):
        try:
            with open(filename, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data.get("Workshop", [])
        except Exception as e:
            print("Error loading workshop prompts:", e)
    return [{
        "name": "Default Workshop Prompt",
        "text": "I need your help with my project. Please provide creative brainstorming and ideas.",
        "provider": "Local",
        "model": "Local Model",
        "max_tokens": 2000,
        "temperature": 1.0
    }]

def load_project_options(project_name):
    """Load project options to inject dynamic values into default prompts."""
    options = {}
    PROJECT_SETTINGS_FILE = "project_settings.json"
    if os.path.exists(PROJECT_SETTINGS_FILE):
        try:
            with open(PROJECT_SETTINGS_FILE, "r", encoding="utf-8") as f:
                all_settings = json.load(f)
            options = all_settings.get(project_name, {})
        except Exception as e:
            print("Error loading project options:", e)
    return options

class PromptsWindow(QDialog):
    def __init__(self, project_name, parent=None):
        super().__init__(parent)
        self.project_name = project_name
        self.setWindowTitle("Prompts - " + project_name)
        self.resize(800, 600)  # Made window larger

        # Load LLM configurations
        # self.llm_configs = load_llm_configs()

        # Define file names
        base_name = f"prompts_{self.project_name.replace(' ', '')}"
        self.prompts_file = f"{base_name}.json"
        self.backup_file = f"{base_name}.bak.json"

        # Default categories
        self.default_categories = ["Workshop", "Summary", "Prose", "Rewrite"]
        self.prompts_data = {}

        self.current_prompt_item = None
        self.selected_model = None  # For persisting selected model in UI

        # NEW: This will store a model name we couldn't set yet
        self.pending_model = None

        self.init_ui()
        self.load_prompts()

    def init_ui(self):
        layout = QVBoxLayout(self)

        # Splitter for tree and editor
        self.splitter = QSplitter(Qt.Horizontal)

        # Left: Tree widget
        tree_widget = QWidget()
        tree_layout = QVBoxLayout(tree_widget)

        self.tree = QTreeWidget()
        self.tree.setHeaderLabel("Prompts")
        self.tree.setContextMenuPolicy(Qt.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self.on_tree_context_menu)
        self.tree.itemClicked.connect(self.on_item_clicked)
        tree_layout.addWidget(self.tree)

        self.splitter.addWidget(tree_widget)

        # Right: Editor with parameters
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)

        # Editor
        self.editor = QTextEdit()
        right_layout.addWidget(self.editor)

        # Parameters panel
        self.parameters_panel = QWidget()
        params_layout = QVBoxLayout(self.parameters_panel)

        # Provider and Model group with refresh button
        model_group = QHBoxLayout()

        # Provider selector
        provider_layout = QVBoxLayout()
        self.provider_label = QLabel("Provider:")
        self.provider_combo = QComboBox()
        self.provider_combo.setMinimumWidth(200)
        provider_layout.addWidget(self.provider_label)
        provider_layout.addWidget(self.provider_combo)
        model_group.addLayout(provider_layout)

        # Model selector with refresh button
        model_layout = QVBoxLayout()
        model_header = QHBoxLayout()
        self.model_label = QLabel("Model:")
        self.refresh_button = QPushButton("↻")  # Refresh symbol
        self.refresh_button.setToolTip("Refresh model list")
        self.refresh_button.setMaximumWidth(30)
        self.refresh_button.clicked.connect(self.refresh_models)
        model_header.addWidget(self.model_label)
        model_header.addWidget(self.refresh_button)
        model_header.addStretch()

        self.model_combo = QComboBox()
        self.model_combo.setMinimumWidth(300)
        # Connect to update internal state immediately if user changes the model manually.
        self.model_combo.currentTextChanged.connect(self.on_model_changed)

        model_layout.addLayout(model_header)
        model_layout.addWidget(self.model_combo)
        model_group.addLayout(model_layout)

        params_layout.addLayout(model_group)

        # Settings group
        settings_group = QHBoxLayout()

        # Max Tokens
        tokens_layout = QVBoxLayout()
        self.max_tokens_label = QLabel("Max Tokens:")
        self.max_tokens_spin = QSpinBox()
        self.max_tokens_spin.setRange(1, 32000)
        self.max_tokens_spin.setValue(2000)
        tokens_layout.addWidget(self.max_tokens_label)
        tokens_layout.addWidget(self.max_tokens_spin)
        settings_group.addLayout(tokens_layout)

        # Temperature
        temp_layout = QVBoxLayout()
        self.temp_label = QLabel("Temperature:")
        self.temp_spin = QDoubleSpinBox()
        self.temp_spin.setRange(0.0, 2.0)
        self.temp_spin.setSingleStep(0.1)
        self.temp_spin.setValue(0.7)
        temp_layout.addWidget(self.temp_label)
        temp_layout.addWidget(self.temp_spin)
        settings_group.addLayout(temp_layout)

        settings_group.addStretch()
        params_layout.addLayout(settings_group)

        # Status label for error messages
        self.status_label = QLabel()
        self.status_label.setStyleSheet("color: red;")
        self.status_label.hide()
        params_layout.addWidget(self.status_label)

        right_layout.addWidget(self.parameters_panel)

        # Hide parameters panel initially
        self.parameters_panel.hide()

        self.splitter.addWidget(right_widget)
        self.splitter.setStretchFactor(1, 2)

        layout.addWidget(self.splitter)

        # Bottom: Save All button
        btn_layout = QHBoxLayout()
        self.save_all_button = QPushButton("Save All")
        self.save_all_button.clicked.connect(self.save_prompts)
        btn_layout.addStretch()
        btn_layout.addWidget(self.save_all_button)
        layout.addLayout(btn_layout)

        # Initialize provider list
        self.update_provider_list()

        self.setLayout(layout)

    def on_model_changed(self, text):
        """Update internal selected model when the combo box selection changes."""
        self.selected_model = text

    def update_provider_list(self):
        """Update the provider combo box with available configurations."""
        self.provider_combo.clear()

        self.llm_configs = WWSettingsManager.get_llm_configs()
        self.active_config = WWSettingsManager.get_active_llm_name()

        # Add each configuration
        for provider, config in self.llm_configs.items():
            display_name = f"{provider} ({config['provider']})"
            self.provider_combo.addItem(display_name, userData=config)
            if provider == self.active_config:
                self.provider_combo.setCurrentText(display_name)


        # Connect after populating to avoid triggering updates
        self.provider_combo.currentTextChanged.connect(self.on_provider_changed)

    def on_provider_changed(self, provider_text):
        """Handle provider selection change."""
        self.refresh_models(True)

    def refresh_models(self, use_cache=False):
        """Refresh the model list for the current provider."""
        current_index = self.provider_combo.currentIndex()
        if current_index < 0:
            return

        config = self.provider_combo.itemData(current_index)
        # If config is a string instead of a dictionary, find the matching config.
        if isinstance(config, str):
            for conf in self.llm_configs:
                display_name = f"{conf['name']} ({conf['provider']})"
                if display_name == config:
                    config = conf
                    break

        if not config:
            return

        if not use_cache:
            self.model_combo.clear()
            self.model_combo.addItem("Loading models...")
            self.model_combo.setEnabled(False)
            self.refresh_button.setEnabled(False)
            QApplication.processEvents()

        # Fetch models using the correct config dictionary.
        try:
            provider_name = config.get("provider")
            provider = WWApiAggregator.aggregator.get_provider(provider_name)
            self.on_models_updated(provider.get_available_models(not use_cache), None)
        except Exception as e:
            self.on_models_updated([], f"Error fetching models: {e}")

    def on_models_updated(self, models, error_msg):
        """Handle updated model list from the fetcher."""
        self.model_combo.clear()
        self.model_combo.addItems(models)
        self.model_combo.addItem("Custom...")

        self.model_combo.setEnabled(True)
        self.refresh_button.setEnabled(True)

        if error_msg:
            self.status_label.setText(error_msg)
            self.status_label.show()
        else:
            self.status_label.hide()

        # If we had a pending model, try to set it now
        if self.pending_model:
            idx = self.model_combo.findText(self.pending_model)
            if idx != -1:
                self.model_combo.setCurrentIndex(idx)
            # Clear the pending model
            self.pending_model = None

    def handle_custom_model(self, model_name):
        """Handle selection of custom model."""
        if model_name == "Custom...":
            custom_model, ok = QInputDialog.getText(
                self, "Custom Model",
                "Enter the model identifier:"
            )
            if ok and custom_model.strip():
                # Temporarily block signals to prevent recursion
                self.model_combo.blockSignals(True)
                # Add and select the custom model
                self.model_combo.insertItem(0, custom_model)
                self.model_combo.setCurrentText(custom_model)
                self.model_combo.blockSignals(False)
            else:
                # If user cancels, revert to first item
                self.model_combo.blockSignals(True)
                self.model_combo.setCurrentIndex(0)
                self.model_combo.blockSignals(False)

    def load_prompts(self):
        """Load prompts and ensure defaults exist."""
        loaded = False
        if os.path.exists(self.prompts_file):
            try:
                with open(self.prompts_file, "r", encoding="utf-8") as f:
                    self.prompts_data = json.load(f)
                loaded = True
            except Exception as e:
                QMessageBox.warning(
                    self, "Error", f"Error loading prompts file: {e}")

        if not loaded and os.path.exists(self.backup_file):
            try:
                with open(self.backup_file, "r", encoding="utf-8") as f:
                    self.prompts_data = json.load(f)
                loaded = True
                QMessageBox.information(
                    self, "Backup Loaded", "Loaded prompts from backup file.")
            except Exception as e:
                QMessageBox.warning(
                    self, "Error", f"Error loading backup prompts file: {e}")

        if not loaded:
            self.prompts_data = {}

        # Ensure categories exist
        for cat in self.default_categories:
            if cat not in self.prompts_data:
                self.prompts_data[cat] = []

        # Load project options for defaults
        options = load_project_options(self.project_name)
        pov = options.get("pov", "First Person")
        pov_char = options.get("pov_character", "the main character")
        tense = options.get("tense", "Past Tense")

        # Default prompt texts
        default_prompts = {
            "Workshop": "I need your help with my project. Please provide creative brainstorming and ideas.",
            "Summary": "Provide a concise summary of the project that retains all relevant information and is optimized for token count.",
            "Prose": f"You are collaborating with the author to write a scene. Write the scene in {{pov}} point of view, from the perspective of {{pov_character}}, and in {{tense}}.",
            "Rewrite": "Rewrite the passage for clarity."
        }

        # Get default provider config
        default_config = self.llm_configs[self.active_config] if self.llm_configs else {
            "name": "Default",
            "provider": "Local",
            "model": "Local Model"
        }

        # Ensure default prompts exist
        for cat in self.default_categories:
            if not self.prompts_data[cat]:
                self.prompts_data[cat].append({
                    "name": f"Default {cat} Prompt",
                    "text": default_prompts.get(cat, ""),
                    "default": True,
                    "provider": default_config["provider"],
                    "model": default_config.get("model", "Local Model"),
                    "max_tokens": 2000,
                    "temperature": 0.7
                })
            else:
                # Update existing prompts with new fields
                for prompt in self.prompts_data[cat]:
                    if "provider" not in prompt:
                        prompt["provider"] = default_config["provider"]
                    if "model" not in prompt:
                        prompt["model"] = default_config.get("model", "Local Model")
                    if "max_tokens" not in prompt:
                        prompt["max_tokens"] = 2000
                    if "temperature" not in prompt:
                        prompt["temperature"] = 0.7

        self.refresh_tree()

    def refresh_tree(self):
        """Rebuild the tree widget from prompts_data."""
        self.tree.clear()
        for category in self.default_categories:
            cat_item = QTreeWidgetItem(self.tree, [category])
            cat_item.setData(0, Qt.UserRole, {"type": "category", "name": category})
            for prompt in self.prompts_data.get(category, []):
                child = QTreeWidgetItem(cat_item, [prompt.get("name", "Unnamed")])
                tooltip = (
                    f"Provider: {prompt.get('provider', 'Local')}\n"
                    f"Model: {prompt.get('model', 'Local Model')}\n"
                    f"Max Tokens: {prompt.get('max_tokens', 2000)}\n"
                    f"Temperature: {prompt.get('temperature', 0.7)}\n"
                    f"Text: {prompt.get('text', '')}"
                )
                if prompt.get("default", False):
                    tooltip += "\n(Default prompt - read-only)"
                child.setToolTip(0, tooltip)
                child.setData(0, Qt.UserRole, {
                    "type": "prompt",
                    "name": prompt.get("name", "Unnamed"),
                    "text": prompt.get("text", ""),
                    "default": prompt.get("default", False),
                    "provider": prompt.get("provider", "Local"),
                    "model": prompt.get("model", "Local Model"),
                    "max_tokens": prompt.get("max_tokens", 2000),
                    "temperature": prompt.get("temperature", 0.7)
                })
        self.tree.expandAll()

        # Start with no item selected
        self.tree.clearSelection()
        self.tree.setCurrentItem(None)

    def on_item_clicked(self, item, column):
        """Handle tree item selection."""
        self.current_prompt_item = item
        data = item.data(0, Qt.UserRole)
        if data and data.get("type") == "prompt":
            self.editor.setPlainText(data.get("text", ""))

            # Find and select the correct provider configuration
            provider = data.get("provider", "Local")
            model = data.get("model", "Local Model")

            # 1) Try to set the provider combo
            for i in range(self.provider_combo.count()):
                config = self.provider_combo.itemData(i)
                if isinstance(config, dict) and config.get("provider") == provider:
                    self.provider_combo.setCurrentIndex(i)
                    break

            # 2) We want to set the model in the combo box, but it might not be loaded yet
            self.selected_model = model
            # Try to find it in the current combo
            idx = self.model_combo.findText(model)
            if idx == -1:
                # Not found, so set a pending_model and refresh
                self.pending_model = model
                self.refresh_models(False)
            else:
                self.model_combo.setCurrentIndex(idx)

            self.max_tokens_spin.setValue(data.get("max_tokens", 2000))
            self.temp_spin.setValue(data.get("temperature", 0.7))
            self.parameters_panel.show()
            self.editor.setReadOnly(data.get("default", False))
        else:
            self.editor.clear()
            self.parameters_panel.hide()
            self.editor.setReadOnly(False)

    def save_prompts(self):
        """Save all prompts to file."""
        # Get the currently selected item directly from the tree widget.
        current_item = self.tree.currentItem()
        prompt_name = None
        if current_item:
            data = current_item.data(0, Qt.UserRole)
            if data and data.get("type") == "prompt" and not data.get("default", False):
                prompt_name = data.get("name")
                new_text = self.editor.toPlainText()
                data["text"] = new_text

                provider_index = self.provider_combo.currentIndex()
                provider_config = self.provider_combo.itemData(provider_index)
                if isinstance(provider_config, dict):
                    data["provider"] = provider_config.get("provider", "Local")
                else:
                    data["provider"] = "Local"

                data["model"] = self.model_combo.currentText()
                data["max_tokens"] = self.max_tokens_spin.value()
                data["temperature"] = self.temp_spin.value()

                tooltip = (
                    f"Provider: {data['provider']}\n"
                    f"Model: {data['model']}\n"
                    f"Max Tokens: {data['max_tokens']}\n"
                    f"Temperature: {data['temperature']}\n"
                    f"Text: {new_text}"
                )
                current_item.setToolTip(0, tooltip)

                parent_item = current_item.parent()
                if parent_item:
                    category = parent_item.text(0)
                    for prompt in self.prompts_data.get(category, []):
                        if prompt.get("name") == data.get("name"):
                            prompt.update(data)
                            break

        try:
            with open(self.prompts_file, "w", encoding="utf-8") as f:
                json.dump(self.prompts_data, f, indent=4)
            with open(self.backup_file, "w", encoding="utf-8") as f:
                json.dump(self.prompts_data, f, indent=4)
            QMessageBox.information(self, "Save All", "Prompts saved successfully.")
            
            # Refresh the tree. This will rebuild all items, so we must avoid holding on to any old references.
            self.refresh_tree()

            # Re-select the prompt so the UI updates properly.
            if prompt_name:
                self.reselect_prompt_by_name(prompt_name)

        except Exception as e:
            QMessageBox.warning(self, "Error", f"Error saving prompts: {e}")

    def reselect_prompt_by_name(self, prompt_name):
        """
        Find the prompt item by name in the tree and re-select it,
        forcing on_item_clicked to update the UI with the new model.
        """
        def find_prompt_item(item):
            if not item:
                return None
            data = item.data(0, Qt.UserRole)
            if data and data.get("type") == "prompt":
                if data.get("name") == prompt_name:
                    return item
            for i in range(item.childCount()):
                child = item.child(i)
                found = find_prompt_item(child)
                if found:
                    return found
            return None

        root = self.tree.invisibleRootItem()
        for i in range(root.childCount()):
            found_item = find_prompt_item(root.child(i))
            if found_item:
                # Trick: temporarily clear the selection, then re-select
                self.tree.setCurrentItem(None)
                self.tree.setCurrentItem(found_item)
                return

    def on_tree_context_menu(self, pos):
        """Handle right-click on tree items."""
        item = self.tree.itemAt(pos)
        if item is None:
            return

        data = item.data(0, Qt.UserRole)
        menu = QMenu()

        if data.get("type") == "category":
            new_action = menu.addAction("New Prompt")
            action = menu.exec_(self.tree.viewport().mapToGlobal(pos))
            if action == new_action:
                self.add_new_prompt(item)
        elif data.get("type") == "prompt":
            if data.get("default", False):
                convert_action = menu.addAction("Convert to Custom")
                info_action = menu.addAction("Default prompt (read-only)")
                action = menu.exec_(self.tree.viewport().mapToGlobal(pos))
                if action == convert_action:
                    data["default"] = False
                    self.editor.setReadOnly(False)
                    QMessageBox.information(
                        self, "Converted", "Prompt is now editable. Remember to save changes.")
                return

            rename_action = menu.addAction("Rename")
            move_up_action = menu.addAction("Move Up")
            move_down_action = menu.addAction("Move Down")
            delete_action = menu.addAction("Delete")
            save_action = menu.addAction("Save")

            action = menu.exec_(self.tree.viewport().mapToGlobal(pos))
            if action == rename_action:
                self.rename_prompt(item)
            elif action == move_up_action:
                self.move_prompt(item, up=True)
            elif action == move_down_action:
                self.move_prompt(item, up=False)
            elif action == delete_action:
                self.delete_prompt(item)
            elif action == save_action:
                self.save_prompt(item)

    def add_new_prompt(self, category_item):
        """Add a new prompt to a category."""
        cat_data = category_item.data(0, Qt.UserRole)
        category_name = cat_data.get("name")
        name, ok = QInputDialog.getText(self, "New Prompt", "Enter prompt name:")
        if not ok or not name.strip():
            return

        name = name.strip()

        # Get current provider config
        provider_index = self.provider_combo.currentIndex()
        provider_config = self.provider_combo.itemData(provider_index)

        new_prompt = {
            "name": name,
            "text": "",
            "default": False,
            "provider": provider_config["provider"] if isinstance(provider_config, dict) else "Local",
            "model": self.model_combo.currentText(),
            "max_tokens": 2000,
            "temperature": 0.7
        }

        self.prompts_data.setdefault(category_name, []).append(new_prompt)
        child = QTreeWidgetItem(category_item, [name])
        child.setData(0, Qt.UserRole, {
            "type": "prompt",
            "name": name,
            "text": "",
            "default": False,
            "provider": new_prompt["provider"],
            "model": new_prompt["model"],
            "max_tokens": 2000,
            "temperature": 0.7
        })
        category_item.setExpanded(True)

    def rename_prompt(self, prompt_item):
        """Rename a prompt."""
        data = prompt_item.data(0, Qt.UserRole)
        if data.get("default", False):
            QMessageBox.information(self, "Rename Prompt",
                                    "Default prompts cannot be renamed.")
            return

        current_name = data.get("name")
        new_name, ok = QInputDialog.getText(
            self, "Rename Prompt", "Enter new prompt name:", text=current_name)
        if ok and new_name.strip():
            new_name = new_name.strip()
            data["name"] = new_name
            prompt_item.setText(0, new_name)
            parent = prompt_item.parent()
            if parent:
                category = parent.text(0)
                for prompt in self.prompts_data.get(category, []):
                    if prompt.get("name") == current_name:
                        prompt["name"] = new_name
                        break

    def move_prompt(self, prompt_item, up=True):
        """Move a prompt up or down in its category."""
        data = prompt_item.data(0, Qt.UserRole)
        if data.get("default", False):
            QMessageBox.information(self, "Move Prompt",
                                    "Default prompts cannot be moved.")
            return

        parent = prompt_item.parent()
        if parent is None:
            return

        index = parent.indexOfChild(prompt_item)
        new_index = index - 1 if up else index + 1
        if new_index < 0 or new_index >= parent.childCount():
            return

        parent.takeChild(index)
        parent.insertChild(new_index, prompt_item)
        category = parent.text(0)
        prompts = self.prompts_data.get(category, [])
        if index < len(prompts) and new_index < len(prompts):
            prompts.insert(new_index, prompts.pop(index))

    def delete_prompt(self, prompt_item):
        """Delete a prompt."""
        data = prompt_item.data(0, Qt.UserRole)
        if data.get("default", False):
            QMessageBox.information(self, "Delete Prompt",
                                    "Default prompts cannot be deleted.")
            return

        name = data.get("name")
        parent = prompt_item.parent()
        if parent is None:
            return

        category = parent.text(0)
        reply = QMessageBox.question(self, "Delete Prompt", f"Delete prompt '{name}'?",
                                     QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            prompts = self.prompts_data.get(category, [])
            for i, prompt in enumerate(prompts):
                if prompt.get("name") == name:
                    prompts.pop(i)
                    break
            parent.removeChild(prompt_item)

# For testing standalone
if __name__ == "__main__":
    from PyQt5.QtWidgets import QApplication
    import sys
    app = QApplication(sys.argv)
    window = PromptsWindow("My Awesome Project")
    window.show()
    sys.exit(app.exec_())
