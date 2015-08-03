<?php
/**
 * Select element
 *
 * @author durso
 */
namespace library\dom\elements\components;
use library\dom\elements\paired;
use library\dom\elements\components\option;


class select extends paired{
    private $options;
    
    public function __construct($name,$multiple = false, $className = "form-control") {
        parent::__construct();
        if($multiple){
            $this->attributes["multiple"] = true;
        }
        $this->attributes["name"] = $name;
        $this->attributes["class"][] = $className;
        $this->tag = "select";
        $this->setId($this->tag);
    }

    public function addOption($option){
        if($option instanceof option){
            $this->addComponent($option);
            $this->options[] = $option;
        } else {
            $opt = new option($option);
            $this->addComponent($opt);
            $this->options[] = $opt;
        }
    }
    
    public function getOptions($index = false){
        if($index === false){
            return $this->options;
        }
        return $this->options[$index];
    }

}