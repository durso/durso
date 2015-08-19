<?php
/**
 * Input element
 *
 * @author durso
 */
namespace library\dom\elements\components;
use library\dom\elements\paired;


class input extends paired{
    
    public function __construct($name = false,$type = "text",$placeholder = false, $className = "form-control") {
        parent::__construct();
        $this->attributes["type"] = $type;
        if($placeholder){
            $this->attributes["placeholder"] = $placeholder;
        }
        if($name){
            $this->attributes["name"] = $name;
        }
        $this->attributes["class"][] = $className;
        $this->tag = "input";
        $this->setId($this->tag);
    }


}