<?php

/**
 * Description of button
 *
 * @author durso
 */
namespace library\layout\elements;
use library\layout\elements\element;
use app\request;

class button extends element {
    protected $type;
    protected $element;
    protected $option;
    
    public function __construct($value, $type = "submit",$option = "default",$tag = "button") {
        $this->value = $value;
        $this->attributes["type"] = $type;
        $this->tag = $tag;
        $this->setCloseTag();
        $this->attributes["class"] = array("btn","btn-".$option);
        $this->setId("button");
    }
    /*
    public function render(){
        $this->html = '<'.$this->tag.$this->addAttribute($this->className,"class").
                $this->addSingleAttribute($this->id, "id").
                $this->addSingleAttribute($this->type, "type").'>'.
                $this->label;
        if($this->tag != "input"){
            $this->html .= '</'.$this->tag.'>';
        }else{
            $this->html .= '>';
        }
        return $this->html;
    }
    */
    public function setCloseTag(){
        if($this->tag != "input"){
            $this->closeTag = true;
        }
    }
   
    
    
}
