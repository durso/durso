<?php
/**
 * Input element
 *
 * @author durso
 */
namespace library\layout\elements;
use library\layout\elements\element;
use library\layout\elements\label;

class input extends element{

    private $label;
    
    public function __construct($type = "text",$placeholder = false, $label = false, $className = "form-control") {
        $this->attributes["type"] = $type;
        $this->attributes["placeholder"] = $placeholder;
        $this->attributes["class"][] = $className;
        $this->tag = "input";
        $this->label = $label;
        $this->setId($this->tag);
    }
    
    public function render(){
        if($this->label){
            $label = new label($this->label,$this->getId());
            $this->html = $label->render();
        }
        return parent::render();
    }

}